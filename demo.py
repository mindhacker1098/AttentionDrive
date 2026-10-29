"""
Presentation demo: watch a trained agent drive through the intersection.

    python demo.py --run runs/multi_attention_s1              # 12-route agent (our extension), live window
    python demo.py --run runs/attention_s1                    # paper's left-turn agent
    python demo.py --run runs/fcn_s1                          # the "blind" baseline, for contrast
    python demo.py --random                                   # random driver
    python demo.py --run runs/multi_attention_s1 --dense      # heavier traffic than in training
    python demo.py --run runs/multi_attention_s1 --video videos/multi_attention.mp4   # save instead of showing
    python demo.py --run runs/multi_attention_s1 --journey    # LONG JOURNEY through a 6-junction city to a finish line

Keys in the live window:  SPACE pause   N next episode   F follow-car / whole-map camera   Q quit

What is on screen
- whole intersection from above; our car is green, its planned route is the green band, its past path the dots
- Ego-Attention agents: a line from our car to each car it is looking at (thicker = more attention,
  amber = head 1, blue = head 2); the 4 most-watched cars get letter tags A-D listed in the panel
- panel: route, speed, time left, Q-value of each action (the chosen one in blue), attention ranking, score
Episodes use the test scenarios from evaluate.py (seeds 10000, 10001, ...).
"""
import argparse
import json
import os

import numpy as np
import pygame
import torch

from common import ACTIONS, LABELS, ROADS, TURNS, EgoAttentionNet, build_model, episode_outcome, make_env
from city_render import CityRenderer
from evaluate import TEST_SEED


class _NoViewer:
    """Stand-in for highway-env's pygame viewer: we render with city_render.py instead."""

    def close(self):
        pass

SIM = 820                       # simulation view, pixels (square)
FOLLOW_SCALE, MAP_SCALE = 9.0, 2.1  # pixels per metre: camera following the car / whole journey city
PANEL_W = 460
HEAD_COLORS = [(255, 176, 32), (64, 160, 255)]       # head 1 amber, head 2 blue
TAG_LETTERS = "ABCD"
DARK, SURFACE, CARD, RULE = (24, 26, 31), (246, 246, 244), (255, 255, 255), (226, 225, 221)
INK, INK_2, INK_3 = (11, 11, 11), (82, 81, 78), (140, 139, 134)
BLUE, GREEN = (42, 120, 214), (50, 200, 0)
OUTCOME = {"arrived": ((0, 131, 0), "MADE IT THROUGH"), "finished": ((0, 131, 0), "FINISH - JOURNEY COMPLETE"), "crashed": ((227, 73, 72), "CRASH"),
           "timeout": ((201, 133, 0), "TOO SLOW - TIME UP")}


class Demo:
    def __init__(self, args):
        self.args = args
        self.task = "left"
        if args.random:
            self.model, self.net = "random", None
            self.task = args.task
        else:
            cfg = json.load(open(os.path.join(args.run, "config.json")))
            self.model, self.task = cfg["model"], cfg.get("task", "left")
            self.net = build_model(self.model, self.task)
            weights = os.path.join(args.run, "model.pt")
            if not os.path.exists(weights):  # training still running: latest checkpoint (saved every 100 episodes)
                weights = os.path.join(args.run, "checkpoint.pt")
                print(f"{args.run} not finished yet - showing the latest checkpoint")
            self.net.load_state_dict(torch.load(weights, map_location="cpu"))
            self.net.eval()
        self.journey = args.journey
        if self.journey:
            import journey
            assert self.task in ("multi", "journey"), "--journey needs a 12-route or city agent (runs/multi_... or journey_...)"
            self.J = journey
            self.env = make_env("attention", task="journey", dense=args.dense)
            junctions = [journey.center(k) for k in range(journey.ROWS * journey.COLS)]
        else:
            self.env = make_env("fcn" if self.net is None else self.model, dense=args.dense, task=self.task)
            junctions = [(0, 0)]
        self.base = self.env.unwrapped
        # we draw everything ourselves (city_render.py); highway-env only has to call us for the in-between frames
        self.base.viewer = _NoViewer()
        self.base._record_video_wrapper = self  # highway-env calls self._capture_frame() 15x per decision
        self.R = CityRenderer(self.base, junctions, view=(SIM, SIM))
        self.cam = None
        self.rng = np.random.default_rng(0)
        self.q, self.action, self.links, self.self_att = None, None, [], None
        self.tally = {"arrived": 0, "crashed": 0, "timeout": 0}
        self.episode, self.banner = 0, None
        self.route_pts, self.trail = [], []
        self.follow = args.follow or args.journey
        self.writer, self.screen, self.clock = None, None, None
        self.paused = self.quit = self.skip = False
        pygame.font.init()
        f = "segoeui,arial"
        self.f_title = pygame.font.SysFont(f, 26, bold=True)
        self.f_h = pygame.font.SysFont(f, 19, bold=True)
        self.f = pygame.font.SysFont(f, 18)
        self.f_s = pygame.font.SysFont(f, 15)
        self.f_big = pygame.font.SysFont(f, 44, bold=True)
        self.f_tag = pygame.font.SysFont(f, 15, bold=True)
        self.f_xs = pygame.font.SysFont(f, 12, bold=True)

    # ------------------------------------------------------------------ decision
    def decide(self, obs):
        if self.net is None:
            self.q, self.action = None, int(self.rng.integers(3))
            return
        x = torch.as_tensor(obs[None])
        with torch.no_grad():
            self.q = self.net(x)[0].numpy()
            if isinstance(self.net, EgoAttentionNet):
                self.match_vehicles(obs, self.net.attend(x)[1][0].numpy())
        self.action = int(self.q.argmax())

    def match_vehicles(self, obs, att):
        """Observation rows are shuffled: find which car each row describes (positions are absolute, /100)."""
        others = [v for v in self.base.road.vehicles if v is not self.base.vehicle]
        links = []
        for row in range(1, obs.shape[0]):
            if obs[row, 0] < 0.5 or not others:
                continue
            car = min(others, key=lambda v: np.linalg.norm(v.position - self.origin() - obs[row, 1:3] * 100))
            links.append((car, att[:, row]))
        self.links = sorted(links, key=lambda l: -l[1].max())
        self.self_att = att[:, 0]

    def origin(self):
        """Positions in the observation are relative to the junction being approached (journey) or the map centre."""
        return self.J.center(self.base.junction) if self.journey else np.zeros(2)

    # ------------------------------------------------------------------ route
    def plan_route(self):
        ego, net = self.base.vehicle, self.base.road.network
        pts = []
        for a, b, i in ego.route:
            lane = net.get_lane((a, b, i or 0))
            s0 = lane.local_coordinates(ego.position)[0] if (a, b) == ego.lane_index[:2] else 0
            pts += [lane.position(s, 0) for s in np.linspace(max(s0, 0), lane.length, 25)]
        self.route_pts, self.trail = pts, []

    def route_label(self):
        if self.journey:
            i = self.base.junction_index
            return f"Junction {i + 1} of {len(self.base.journey)}", self.J.TURN_NAMES[self.base.journey[i][3]]
        if self.task == "multi":
            return ROADS[self.base.start_road].capitalize() + " road", TURNS[self.base.turn]
        return "South road", "LEFT"

    # ------------------------------------------------------------------ drawing
    def _capture_frame(self):
        if self.quit or self.skip:
            return
        ego = self.base.vehicle
        if self.follow:  # camera glides after our car, looking a bit ahead
            target = ego.position + 10 * np.array([np.cos(ego.heading), np.sin(ego.heading)])
            self.cam = target if self.cam is None else self.cam + 0.15 * (target - self.cam)
            scale = FOLLOW_SCALE
        elif self.journey:  # whole city
            self.cam = self.J.center(0) + np.array([self.J.SPACING, self.J.SPACING / 2])
            scale = MAP_SCALE
        else:  # whole intersection
            self.cam, scale = np.zeros(2), SIM / 120
        self.trail.append(ego.position.copy())
        R = self.R

        def under(surf):
            self.draw_route(surf, R)
            if self.journey:
                self.draw_finish(surf, R)

        sim = R.frame(self.cam, scale, under=under, over=lambda surf: self.draw_attention(surf, R))
        if self.journey and self.follow:
            self.draw_minimap(sim)
        if self.banner:
            self.draw_banner(sim)
        full = pygame.Surface((SIM + PANEL_W, SIM))
        full.fill(SURFACE)
        full.blit(sim, (0, 0))
        self.draw_panel(full, SIM)
        self.output(full)

    def draw_route(self, surf, world):
        layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        if len(self.route_pts) > 1:
            pygame.draw.lines(layer, (*GREEN, 80), False, [world.vec2pix(p) for p in self.route_pts],
                              max(6, int(3.4 * world.scale)))
            end = world.vec2pix(self.route_pts[-1])
            pygame.draw.circle(layer, (*GREEN, 160), end, 9)
            pygame.draw.circle(layer, (255, 255, 255, 220), end, 4)
        for p in self.trail[::3]:
            pygame.draw.circle(layer, (255, 255, 255, 150), world.vec2pix(p), 2)
        surf.blit(layer, (0, 0))

    def draw_attention(self, surf, world):
        if not self.links:
            return
        layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        ego = world.vec2pix(self.base.vehicle.position)
        tagged = {id(car) for car, w in self.links[:4] if w.max() >= 0.04}
        for h, color in enumerate(HEAD_COLORS):
            for car, w in self.links:
                if w[h] > 0.04:
                    target = world.vec2pix(car.position)
                    pygame.draw.line(layer, (*color, 190), ego, target, max(2, int(round(w[h] * 14))))
                    if id(car) in tagged:
                        pygame.draw.circle(layer, (*color, 220), target, int(world.scale * 3.2) + 4 * h, 2)
            if self.self_att is not None and self.self_att[h] > 0.04:  # attention on itself = "road is clear"
                pygame.draw.circle(layer, (*color, 120), ego, int(world.scale * 4) + 6 * h,
                                   max(2, int(round(self.self_att[h] * 5))))
        surf.blit(layer, (0, 0))
        for letter, (car, w) in zip(TAG_LETTERS, self.links):
            if w.max() < 0.04:
                break
            x, y = world.vec2pix(car.position)
            off = int(world.scale * 3.4)
            pygame.draw.circle(surf, DARK, (x + off, y - off), 11)
            pygame.draw.circle(surf, (255, 255, 255), (x + off, y - off), 11, 1)
            t = self.f_tag.render(letter, True, (255, 255, 255))
            surf.blit(t, t.get_rect(center=(x + off, y - off)))

    def draw_finish(self, surf, world):
        """Chequered finish line across the last road, where the journey counts as complete."""
        lane = self.base.road.network.get_lane((*self.base.final_lane, 0))
        p, h = lane.position(25, 0), lane.heading_at(25)
        across = np.array([-np.sin(h), np.cos(h)])
        along = np.array([np.cos(h), np.sin(h)])
        for i in range(-4, 4):
            for j in range(2):
                c = (255, 255, 255) if (i + j) % 2 else (20, 20, 20)
                corners = [p + across * i + along * j, p + across * (i + 1) + along * j,
                           p + across * (i + 1) + along * (j + 1), p + across * i + along * (j + 1)]
                pygame.draw.polygon(surf, c, [world.vec2pix(q) for q in corners])
        x, y = world.vec2pix(p + across * 7)
        t = self.f_tag.render("FINISH", True, (255, 255, 255))
        pygame.draw.rect(surf, (20, 20, 20), t.get_rect(center=(x, y)).inflate(10, 6), border_radius=4)
        surf.blit(t, t.get_rect(center=(x, y)))

    def draw_minimap(self, surf):
        """Whole city in the top-left corner: real map thumbnail, our route (green), other cars, us, the finish."""
        img, to_map = self.R.minimap((260, 190))
        ox, oy = 12, 12
        to_px = lambda p: (to_map(p)[0] + ox, to_map(p)[1] + oy)
        pygame.draw.rect(surf, DARK, (ox - 4, oy - 4, img.get_width() + 8, img.get_height() + 26), border_radius=8)
        surf.blit(img, (ox, oy))
        H = img.get_height() + 26
        if len(self.route_pts) > 1:
            pygame.draw.lines(surf, GREEN, False, [to_px(p) for p in self.route_pts], 3)
        for v in self.base.road.vehicles:
            if v is not self.base.vehicle:
                pygame.draw.circle(surf, (255, 255, 255), to_px(v.position), 2)
        lane = self.base.road.network.get_lane((*self.base.final_lane, 0))
        fx, fy = to_px(lane.position(25, 0))
        pygame.draw.rect(surf, (255, 255, 255), (fx - 4, fy - 4, 8, 8))
        pygame.draw.circle(surf, (255, 255, 255), to_px(self.base.vehicle.position), 6)
        pygame.draw.circle(surf, GREEN, to_px(self.base.vehicle.position), 4)
        self.text(surf, "CITY MAP  ·  green = our route  ·  square = finish", (ox + 2, oy + H - 22), self.f_xs,
                  (190, 192, 198))

    def draw_banner(self, surf):
        color, label = OUTCOME[self.banner]
        layer = pygame.Surface((SIM, 90), pygame.SRCALPHA)
        layer.fill((*color, 225))
        surf.blit(layer, (0, SIM // 2 - 45))
        t = self.f_big.render(label, True, (255, 255, 255))
        surf.blit(t, t.get_rect(center=(SIM // 2, SIM // 2)))

    def text(self, surf, s, pos, font=None, color=INK, right=False):
        t = (font or self.f).render(s, True, color)
        surf.blit(t, t.get_rect(topright=pos) if right else pos)
        return t.get_width()

    def card(self, surf, x, y, w, h, title):
        pygame.draw.rect(surf, CARD, (x, y, w, h), border_radius=10)
        pygame.draw.rect(surf, RULE, (x, y, w, h), 1, border_radius=10)
        self.text(surf, title, (x + 14, y + 10), self.f_h)

    def arrow(self, surf, turn, cx, cy, color):
        """Small manoeuvre icon: up-arrow that bends left/right."""
        pygame.draw.line(surf, color, (cx, cy + 14), (cx, cy - 2), 5)
        if turn == "STRAIGHT":
            pygame.draw.polygon(surf, color, [(cx - 8, cy - 2), (cx + 8, cy - 2), (cx, cy - 14)])
            return
        d = -1 if turn == "LEFT" else 1
        pygame.draw.line(surf, color, (cx, cy - 2), (cx + 10 * d, cy - 2), 5)
        pygame.draw.polygon(surf, color, [(cx + 10 * d, cy - 10), (cx + 10 * d, cy + 6), (cx + 20 * d, cy - 2)])

    def draw_panel(self, surf, x0):
        pygame.draw.rect(surf, DARK, (x0, 0, PANEL_W, 74))
        self.text(surf, "Social Attention  ·  DQN", (x0 + 20, 10), self.f_title, (255, 255, 255))
        sub = f"{LABELS[self.model]} agent  ·  " + {"multi": "12 routes (our extension)",
                                                     "journey": "fine-tuned on the city",
                                                     "left": "left turn (paper's task)"}[self.task]
        if self.journey and self.task == "multi":
            sub = f"{LABELS[self.model]} agent  ·  12-route agent, never trained on the city"
        self.text(surf, sub, (x0 + 20, 44), self.f_s, (190, 192, 198))
        x, w, y = x0 + 16, PANEL_W - 32, 88

        # route + status
        if self.journey:
            self.journey_card(surf, x, y, w)
        else:
            self.route_card(surf, x, y, w)
        self.status_cards(surf, x, y + 130, w)

    def route_card(self, surf, x, y, w):
        self.card(surf, x, y, w, 118, "Route")
        road, turn = self.route_label()
        self.arrow(surf, turn, x + 30, y + 64, BLUE)
        self.text(surf, f"{road}  ->  turn {turn}" if turn != "STRAIGHT" else f"{road}  ->  go STRAIGHT",
                  (x + 56, y + 50), self.f_h, INK)
        traffic = "dense traffic" if self.args.dense else "normal traffic"
        self.text(surf, f"Episode {self.episode}/{self.args.episodes}  ·  {traffic}", (x + 56, y + 76), self.f_s, INK_2)

    def journey_card(self, surf, x, y, w):
        """Turn sequence of the whole journey (done = green, now = blue, later = grey) + progress bar."""
        self.card(surf, x, y, w, 118, f"Journey {self.episode}/{self.args.episodes}  -  to the finish line")
        i_now = self.base.junction_index
        for i, (*_, turn) in enumerate(self.base.journey):
            cx = x + 30 + i * 62
            color = (0, 131, 0) if i < i_now else BLUE if i == i_now else INK_3
            self.arrow(surf, self.J.TURN_NAMES[turn], cx, y + 56, color)
            if i < len(self.base.journey) - 1:
                pygame.draw.line(surf, RULE, (cx + 24, y + 56), (cx + 38, y + 56), 2)
        flag_x = x + 30 + len(self.base.journey) * 62 - 8
        for r in range(3):
            for c in range(3):
                pygame.draw.rect(surf, INK if (r + c) % 2 else CARD, (flag_x + 5 * c, y + 44 + 5 * r, 5, 5))
        label, turn = self.route_label()
        self.text(surf, f"{label}:  {'go ' if turn == 'STRAIGHT' else 'turn '}{turn}", (x + 14, y + 74), self.f_h)
        prog = self.base.progress()
        pygame.draw.rect(surf, RULE, (x + 14, y + 100, w - 90, 8), border_radius=4)
        pygame.draw.rect(surf, (0, 131, 0), (x + 14, y + 100, max(6, int(prog * (w - 90))), 8), border_radius=4)
        self.text(surf, f"{100 * prog:.0f}%", (x + w - 14, y + 94), self.f_s, INK_2, right=True)

    def status_cards(self, surf, x, y, w):
        speed, t, dur = self.base.vehicle.speed, self.base.time, self.base.config["duration"]
        self.card(surf, x, y, w, 92, "Speed and time")
        bw = w - 150
        for i, (label, frac, val) in enumerate([("speed", min(speed / 10, 1), f"{speed:4.1f} m/s"),
                                                ("time", min(t / dur, 1), f"{t:4.1f} / {dur} s")]):
            yy = y + 40 + 24 * i
            self.text(surf, label, (x + 14, yy - 2), self.f_s, INK_2)
            pygame.draw.rect(surf, RULE, (x + 64, yy + 3, bw, 12), border_radius=6)
            pygame.draw.rect(surf, BLUE if i == 0 else INK_3, (x + 64, yy + 3, max(6, int(frac * bw)), 12),
                             border_radius=6)
            self.text(surf, val, (x + w - 12, yy - 2), self.f_s, INK_2, right=True)
        y += 104

        # decision
        self.card(surf, x, y, w, 130, "Decision  (Q-value = expected future reward)"
                  if self.q is not None else "Decision  (random driver)")
        q = self.q if self.q is not None else np.zeros(3)
        lo, hi = min(q.min(), 0), max(q.max(), 1e-6)
        for a, name in enumerate(ACTIONS):
            yy, chosen = y + 42 + 27 * a, a == self.action
            self.text(surf, name, (x + 14, yy), self.f_h if chosen else self.f, INK if chosen else INK_2)
            pygame.draw.rect(surf, RULE, (x + 110, yy + 5, w - 190, 14), border_radius=7)
            if self.q is not None:
                frac = (q[a] - lo) / (hi - lo + 1e-9)
                pygame.draw.rect(surf, BLUE if chosen else (170, 168, 162),
                                 (x + 110, yy + 5, max(8, int(frac * (w - 190))), 14), border_radius=7)
                self.text(surf, f"{q[a]:.2f}", (x + w - 12, yy), self.f_s, INK_2, right=True)
        y += 142

        # attention ranking
        if isinstance(self.net, EgoAttentionNet):
            self.card(surf, x, y, w, 196, "Who is it watching?")
            hx = x + w - 190
            for h, c in enumerate(HEAD_COLORS):
                pygame.draw.line(surf, c, (hx + 92 * h, y + 21), (hx + 92 * h + 22, y + 21), 5)
                self.text(surf, f"head {h + 1}", (hx + 92 * h + 28, y + 11), self.f_s, INK_2)
            rows = [(TAG_LETTERS[i], w_) for i, (_, w_) in enumerate(self.links[:4]) if w_.max() >= 0.04]
            if self.self_att is not None:
                rows.append(("self", self.self_att))
            for i, (tag, wt) in enumerate(rows[:5]):
                yy = y + 46 + 28 * i
                self.text(surf, f"car {tag}" if tag != "self" else "itself", (x + 14, yy), self.f, INK_2)
                for h, c in enumerate(HEAD_COLORS):
                    bx = x + 90 + h * ((w - 110) // 2)
                    bw2 = (w - 130) // 2
                    pygame.draw.rect(surf, RULE, (bx, yy + 6, bw2, 12), border_radius=6)
                    pygame.draw.rect(surf, c, (bx, yy + 6, max(4, int(wt[h] * bw2)), 12), border_radius=6)
            if not rows:
                self.text(surf, "(no cars nearby)", (x + 14, y + 50), self.f, INK_3)
            y += 208

        # score
        self.card(surf, x, y, w, 66, "Score so far")
        cx = x + 14
        for k, label in [("arrived", "made it"), ("crashed", "crashed"), ("timeout", "too slow")]:
            pygame.draw.circle(surf, OUTCOME[k][0], (cx + 7, y + 47), 7)
            cx += 22 + self.text(surf, f"{label}: {self.tally[k]}", (cx + 18, y + 37), self.f, INK) + 18
        y += 78
        legend = ("green band = planned route  ·  dots = path so far  ·  lines = attention"
                  if isinstance(self.net, EgoAttentionNet) else "green band = planned route  ·  dots = path so far")
        self.text(surf, legend, (x + 4, y), self.f_s, INK_3)
        if not self.args.video:
            self.text(surf, "SPACE pause  ·  N next  ·  F camera  ·  Q quit", (x + 4, y + 22), self.f_s, INK_3)

    # ------------------------------------------------------------------ output
    def output(self, surf):
        if self.args.video:
            if self.writer is None:
                import imageio
                os.makedirs(os.path.dirname(self.args.video) or ".", exist_ok=True)
                self.writer = imageio.get_writer(self.args.video, fps=self.args.fps, macro_block_size=1, quality=8)
            self.writer.append_data(pygame.surfarray.array3d(surf).swapaxes(0, 1))
            return
        if self.screen is None:
            pygame.display.init()
            self.screen = pygame.display.set_mode(surf.get_size())
            pygame.display.set_caption("Social Attention - intersection demo")
            self.clock = pygame.time.Clock()
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    self.quit = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                    self.skip = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    self.follow = not self.follow
                    self.cam = None
            if not self.paused or self.quit or self.skip:
                break
            self.clock.tick(30)
        self.clock.tick(self.args.fps)

    def run(self):
        for i in range(self.args.episodes):
            if self.quit:
                break
            self.episode, self.banner, self.skip = i + 1, None, False
            obs, _ = self.env.reset(seed=TEST_SEED + self.args.start + i)
            self.links, self.self_att, self.q, self.action = [], None, None, None
            self.plan_route()
            done = False
            while not done and not self.quit and not self.skip:
                self.decide(obs)
                self._capture_frame()
                self.base.enable_auto_render = True  # ask highway-env for the 14 in-between frames too
                obs, _, terminated, truncated, info = self.env.step(self.action)
                done = terminated or truncated
            if self.skip or self.quit:
                continue
            outcome = episode_outcome(self.env, terminated, info["crashed"])
            self.tally[outcome] += 1
            self.banner = "finished" if (outcome == "arrived" and self.journey) else outcome
            for _ in range(self.args.fps):  # hold the result for one second
                self._capture_frame()
        if self.writer:
            self.writer.close()
            print(f"-> {self.args.video}")
        print("results:", self.tally)
        self.env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/multi_attention_s1")
    p.add_argument("--random", action="store_true", help="random driver instead of a trained agent")
    p.add_argument("--task", default="multi", choices=["left", "multi"], help="task for --random")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--start", type=int, default=0, help="first test scenario")
    p.add_argument("--dense", action="store_true", help="heavier traffic than in training")
    p.add_argument("--follow", action="store_true", help="camera follows our car instead of showing the whole junction")
    p.add_argument("--journey", action="store_true", help="long journey through a 6-junction city (needs a multi_ agent)")
    p.add_argument("--video", default=None, help="save an mp4 instead of opening a window")
    p.add_argument("--fps", type=int, default=15)
    Demo(p.parse_args()).run()


if __name__ == "__main__":
    main()
