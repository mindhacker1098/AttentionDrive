"""
A nicer top-down renderer for the demo (highway-env only draws grey boxes).

Static world, drawn once per map: grass, sidewalks, asphalt, lane markings, stop lines, zebra crossings,
buildings with shadows and roof details, trees.
Every frame: cars as sprites with windows, headlights, brake lights (when slowing down) and blinking
turn signals (from each car's planned route - the same "intention" the agent gets as input), crashed cars
in red with a warning marker, and our car highlighted with a glow.

    r = CityRenderer(env.unwrapped, junctions=[(0, 0)], view=(820, 820))
    surf = r.frame(camera_center, scale, under=draw_route_fn, over=draw_attention_fn)
    r.vec2pix(world_point) -> pixel  (valid for the last frame)
"""
import math

import numpy as np
import pygame
from highway_env.road.lane import CircularLane, LineType, StraightLane

GRASS = (92, 132, 74)
SIDEWALK = (196, 194, 186)
CURB = (160, 158, 150)
ASPHALT = (60, 63, 69)
MARK = (235, 235, 230)
ROOFS = [(178, 170, 160), (152, 160, 176), (192, 176, 152), (141, 152, 140), (204, 192, 182), (122, 128, 140),
         (168, 132, 118), (110, 134, 150)]
CAR_COLORS = [(52, 110, 196), (196, 58, 52), (230, 230, 228), (40, 42, 48), (232, 184, 48), (120, 124, 132),
              (64, 150, 118), (150, 70, 160), (210, 120, 50), (90, 160, 210)]
EGO_COLOR = (40, 200, 90)
BASE_SCALE = 9.0  # pixels per metre of the pre-rendered static world


def lane_points(lane, lateral=0.0, step=1.0, s0=0.0, s1=None):
    s1 = lane.length if s1 is None else s1
    n = max(2, int((s1 - s0) / step) + 1)
    return [lane.position(s, lateral) for s in np.linspace(s0, s1, n)]


class CityRenderer:
    def __init__(self, base_env, junctions, view=(820, 820), seed=7):
        self.env, self.junctions, self.view = base_env, [np.array(j, float) for j in junctions], view
        self.rng = np.random.default_rng(seed)
        self.static, self.signature, self.sprites = None, None, {}
        self.cam, self.scale = np.zeros(2), BASE_SCALE
        self.frame_no = 0
        self.colors = {}
        self.scaled_cache = {}

    # ------------------------------------------------------------------ static world
    def lanes(self):
        net = self.env.road.network
        return [l for tos in net.graph.values() for ls in tos.values() for l in ls]

    def ensure_static(self):
        lanes = self.lanes()
        sig = (len(lanes), tuple(np.round(lanes[0].position(0, 0), 1)), tuple(np.round(lanes[-1].position(0, 0), 1)))
        if sig == self.signature:
            return
        self.signature = sig
        pts = np.array([p for l in lanes for p in (l.position(0, 0), l.position(l.length, 0))])
        self.lo = pts.min(0) - 40
        self.hi = pts.max(0) + 40
        size = ((self.hi - self.lo) * BASE_SCALE).astype(int)
        S = BASE_SCALE
        P = lambda p: (int((p[0] - self.lo[0]) * S), int((p[1] - self.lo[1]) * S))
        surf = pygame.Surface(size)
        self.grass(surf)

        straight = [l for l in lanes if isinstance(l, StraightLane)]
        right_turns = [l for l in lanes if isinstance(l, CircularLane) and l.radius < 10]
        # sidewalks (wide grey band) under straight roads and right-turn corners
        mask_surf = pygame.Surface(size)
        for target, color, extra in [(surf, CURB, 3.2), (surf, SIDEWALK, 2.8), (mask_surf, (255, 255, 255), 4.5)]:
            for l in straight + right_turns:
                w = int((l.width + 2 * extra) * S)
                pygame.draw.lines(target, color, False, [P(p) for p in lane_points(l, 0, 2.0)], w)
                for p in (l.position(0, 0), l.position(l.length, 0)):
                    pygame.draw.circle(target, color, P(p), w // 2)
        # asphalt for every lane + a solid box in each junction
        for l in lanes:
            pygame.draw.lines(surf, ASPHALT, False, [P(p) for p in lane_points(l, 0, 1.0)], int((l.width + 0.4) * S))
        for c in self.junctions:
            half = 6.5
            pygame.draw.rect(surf, ASPHALT, (*P(c - half), int(2 * half * S), int(2 * half * S)))
        # markings on straight roads
        for l in straight:
            for side, lt in zip((-1, 1), l.line_types):
                lat = side * l.width / 2
                if lt == LineType.CONTINUOUS:
                    pygame.draw.lines(surf, MARK, False, [P(p) for p in lane_points(l, lat * 0.92, 2.0)], 2)
                elif lt == LineType.STRIPED:
                    for s in np.arange(1.5, l.length - 1.5, 6.0):
                        pygame.draw.line(surf, MARK, P(l.position(s, lat)), P(l.position(min(s + 3, l.length), lat)), 2)
        # stop lines + zebra crossings at every junction arm (incoming lanes end near the junction)
        for l in straight:
            end = l.position(l.length, 0)
            if not any(np.linalg.norm(end - c) < 14 for c in self.junctions):
                continue
            h = l.heading_at(l.length)
            along, across = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
            stop = end - along * 0.6
            pygame.draw.line(surf, MARK, P(stop - across * 1.9), P(stop + across * 1.9), int(0.45 * S))
            for k in np.arange(-1.8, 1.9, 0.9):      # zebra stripes over this lane, 2-5 m before the stop line
                a = end - along * 5.0 + across * k
                pygame.draw.line(surf, MARK, P(a), P(a + along * 3.0), int(0.45 * S))
        self.buildings(surf, mask_surf, size)
        self.static = surf
        self.scaled_cache = {}

    def grass(self, surf):
        tile = pygame.Surface((128, 128))
        noise = self.rng.integers(-9, 10, (128, 128))
        arr = np.clip(np.array(GRASS)[None, None, :] + noise[..., None] * np.array([0.6, 1.0, 0.5]), 0, 255)
        pygame.surfarray.blit_array(tile, arr.astype(np.uint8))
        for x in range(0, surf.get_width(), 128):
            for y in range(0, surf.get_height(), 128):
                surf.blit(tile, (x, y))

    def buildings(self, surf, mask_surf, size):
        """Random buildings and trees on free land (not on roads/sidewalks, not overlapping each other)."""
        S = BASE_SCALE
        road = pygame.mask.from_threshold(mask_surf, (255, 255, 255), (1, 1, 1, 255))
        taken = road.copy()
        W, H = size
        for _ in range(900):
            w, h = self.rng.uniform(9, 22, 2) * S
            x, y = self.rng.uniform(0, W - w), self.rng.uniform(0, H - h)
            m = pygame.mask.Mask((int(w + 2 * S), int(h + 2 * S)), fill=True)
            if taken.overlap(m, (int(x - S), int(y - S))):
                continue
            taken.draw(m, (int(x - S), int(y - S)))
            roof = ROOFS[self.rng.integers(len(ROOFS))]
            shadow = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)
            shadow.fill((0, 0, 0, 70))
            surf.blit(shadow, (x + 0.9 * S, y + 0.9 * S))
            pygame.draw.rect(surf, roof, (x, y, w, h), border_radius=3)
            pygame.draw.rect(surf, tuple(int(c * 0.78) for c in roof), (x, y, w, h), 3, border_radius=3)
            for _ in range(self.rng.integers(0, 3)):           # roof units
                uw, uh = self.rng.uniform(1.5, 3.0, 2) * S
                ux, uy = x + self.rng.uniform(0.15, 0.7) * w, y + self.rng.uniform(0.15, 0.7) * h
                pygame.draw.rect(surf, tuple(min(255, int(c * 1.12)) for c in roof), (ux, uy, uw, uh), border_radius=2)
        for _ in range(1400):                                    # trees in the gaps
            r = self.rng.uniform(1.6, 3.2) * S
            x, y = self.rng.uniform(r, W - r), self.rng.uniform(r, H - r)
            m = pygame.mask.Mask((int(2 * r), int(2 * r)), fill=True)
            if taken.overlap(m, (int(x - r), int(y - r))):
                continue
            taken.draw(m, (int(x - r), int(y - r)))
            pygame.draw.circle(surf, (40, 70, 36), (int(x + 0.5 * S), int(y + 0.5 * S)), int(r))
            pygame.draw.circle(surf, (58, 104, 50), (int(x), int(y)), int(r))
            pygame.draw.circle(surf, (84, 136, 66), (int(x - r * 0.3), int(y - r * 0.3)), int(r * 0.55))

    # ------------------------------------------------------------------ cars
    def car_sprite(self, color, scale, ego=False):
        key = (color, round(scale, 2), ego)
        if key in self.sprites:
            return self.sprites[key]
        L, W, k = 5.0, 2.0, 4  # draw at 4x then scale down for smooth edges
        px = scale * k
        w, h = int(L * px), int(W * px)
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        r = int(0.5 * px)
        pygame.draw.rect(s, color, (0, 0, w, h), border_radius=r)
        dark = tuple(int(c * 0.55) for c in color)
        pygame.draw.rect(s, dark, (0, 0, w, h), max(1, int(0.12 * px)), border_radius=r)
        glass = (150, 186, 214) if not ego else (190, 235, 205)
        pygame.draw.rect(s, glass, (int(0.62 * w), int(0.14 * h), int(0.16 * w), int(0.72 * h)), border_radius=int(0.2 * px))
        pygame.draw.rect(s, glass, (int(0.12 * w), int(0.18 * h), int(0.1 * w), int(0.64 * h)), border_radius=int(0.2 * px))
        pygame.draw.rect(s, tuple(min(255, int(c * 1.15)) for c in color),
                         (int(0.25 * w), int(0.18 * h), int(0.35 * w), int(0.64 * h)), border_radius=int(0.2 * px))
        if ego:
            pygame.draw.rect(s, (255, 255, 255), (int(0.25 * w), int(0.44 * h), int(0.35 * w), int(0.12 * h)))
        sprite = pygame.transform.smoothscale(s, (max(4, w // k), max(2, h // k)))
        self.sprites[key] = sprite
        return sprite

    def color_of(self, v):
        if v not in self.colors:
            self.colors[v] = CAR_COLORS[len(self.colors) % len(CAR_COLORS)]
        return self.colors[v]

    @staticmethod
    def next_turn(v):
        """+1 left, -1 right, 0 straight/none: from the car's current lane and planned route (its 'blinker')."""
        for a, b, _ in [v.lane_index] + list(v.route or []):
            if a.startswith("ir") and b.startswith("il"):
                d = (int(b[2]) - int(a[2])) % 4
                return {1: 1, 3: -1}.get(d, 0)
        return 0

    def draw_car(self, surf, v, ego=False):
        S = self.scale
        x, y = self.vec2pix(v.position)
        if not (-40 < x < self.view[0] + 40 and -40 < y < self.view[1] + 40):
            return
        color = EGO_COLOR if ego else self.color_of(v)
        if ego:
            glow = pygame.Surface((int(9 * S), int(9 * S)), pygame.SRCALPHA)
            pygame.draw.circle(glow, (80, 255, 140, 60), (glow.get_width() // 2,) * 2, int(4.4 * S))
            surf.blit(glow, glow.get_rect(center=(x, y)))
        ang = -math.degrees(v.heading)
        h = v.heading
        along, across = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
        # shadow
        sh = pygame.Surface((int(5.2 * S), int(2.4 * S)), pygame.SRCALPHA)
        pygame.draw.rect(sh, (0, 0, 0, 80), sh.get_rect(), border_radius=int(0.6 * S))
        sh = pygame.transform.rotate(sh, ang)
        surf.blit(sh, sh.get_rect(center=(x + 0.35 * S, y + 0.45 * S)))
        sprite = pygame.transform.rotate(self.car_sprite(color, S, ego), ang)
        surf.blit(sprite, sprite.get_rect(center=(x, y)))
        p = v.position
        lamp = max(1, int(0.32 * S))
        for side in (-0.65, 0.65):
            pygame.draw.circle(surf, (255, 246, 200), self.vec2pix(p + along * 2.35 + across * side), lamp)
        braking = v.action.get("acceleration", 0) < -0.6 or v.speed < 0.3
        tail = (255, 40, 40) if braking else (150, 30, 30)
        for side in (-0.7, 0.7):
            pygame.draw.circle(surf, tail, self.vec2pix(p - along * 2.4 + across * side), lamp + (1 if braking else 0))
        turn = self.next_turn(v)
        if turn and (self.frame_no // 6) % 2 == 0:  # blinking amber indicators front + back on that side
            side = -1.0 if turn > 0 else 1.0  # left = -y side of the car in highway-env coordinates
            for f in (2.2, -2.2):
                pygame.draw.circle(surf, (255, 170, 20), self.vec2pix(p + along * f + across * side), lamp + 1)
        if v.crashed:
            pygame.draw.circle(surf, (227, 73, 72), (x, y), int(2.8 * S), max(2, int(0.35 * S)))
            pygame.draw.polygon(surf, (255, 210, 0), [(x, y - int(5.6 * S)), (x - int(1.5 * S), y - int(3.2 * S)),
                                                      (x + int(1.5 * S), y - int(3.2 * S))])
            pygame.draw.line(surf, (20, 20, 20), (x, y - int(5.0 * S)), (x, y - int(4.0 * S)), 2)

    # ------------------------------------------------------------------ frame
    def vec2pix(self, p):
        return (int((p[0] - self.cam[0]) * self.scale + self.view[0] / 2),
                int((p[1] - self.cam[1]) * self.scale + self.view[1] / 2))

    def static_at(self, scale):
        if abs(scale - BASE_SCALE) < 1e-6:
            return self.static
        key = round(scale, 3)
        if key not in self.scaled_cache:
            size = (int(self.static.get_width() * scale / BASE_SCALE), int(self.static.get_height() * scale / BASE_SCALE))
            self.scaled_cache[key] = pygame.transform.smoothscale(self.static, size)
        return self.scaled_cache[key]

    def frame(self, camera, scale, under=None, over=None):
        self.ensure_static()
        self.frame_no += 1
        self.cam, self.scale = np.asarray(camera, float), scale
        out = pygame.Surface(self.view)
        out.fill(GRASS)
        world = self.static_at(scale)
        out.blit(world, self.vec2pix(self.lo))
        if under:
            under(out)
        ego = self.env.vehicle
        for v in self.env.road.vehicles:
            if v is not ego:
                self.draw_car(out, v)
        self.draw_car(out, ego, ego=True)
        if over:
            over(out)
        return out

    def minimap(self, size):
        """The static world shrunk to fit `size` (cached) and the world->minimap transform."""
        self.ensure_static()
        sc = min(size[0] / (self.hi - self.lo)[0], size[1] / (self.hi - self.lo)[1])
        key = ("mini", size)
        if key not in self.scaled_cache:
            self.scaled_cache[key] = pygame.transform.smoothscale(
                self.static, (int((self.hi - self.lo)[0] * sc), int((self.hi - self.lo)[1] * sc)))
        return self.scaled_cache[key], lambda p: (int((p[0] - self.lo[0]) * sc), int((p[1] - self.lo[1]) * sc))
