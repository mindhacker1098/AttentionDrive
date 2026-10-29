"""
Long journey through a small city: a 2 x 3 grid of the paper's intersection, connected by roads.

Our car gets a random journey from one edge of the map to a FINISH LINE on another edge, passing
3-5 junctions with several turns, through traffic everywhere. Like a real self-driving stack, it is split in two:

  1. route planner (like Google Maps): the shortest path through the grid fixes which way to turn at each junction
  2. driving brain: our Ego-Attention DQN from the 12-route extension decides SLOWER / IDLE / FASTER every second

The trained agent only ever saw ONE intersection. At every junction we show it the scene the way it saw it
during training: positions relative to the junction it is approaching, its goal = the exit it must take there,
and the "turn signal" (next exit) of every other car. So the skill learned at one junction is reused at all of them.

    env = make_journey_env()           # gymnasium env, observation (15, 9) like the multi-route task
"""
import numpy as np
import gymnasium as gym
from highway_env.envs.intersection_env import IntersectionEnv
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.utils import class_from_path
from highway_env.vehicle.kinematics import Vehicle

class JourneyNetwork(RoadNetwork):
    """Marker class: road network of the journey map (enables the fast lane lookup below)."""


_original_on_state_update = Vehicle.on_state_update


def _fast_on_state_update(self):
    """
    highway-env finds a car's lane by checking EVERY lane of the map, every frame (120 lanes here -> very slow).
    Cars only ever move forward along connected lanes, so on the journey map we only check the current lane
    and the lanes that start where it ends. Same result, ~20x less work. Other maps keep the original code.
    """
    road = self.road
    if not road or not isinstance(road.network, JourneyNetwork) or getattr(self, "lane_index", None) is None:
        return _original_on_state_update(self)
    net = road.network
    a, b, i = self.lane_index
    candidates = [(a, b, i)] + [(b, c, 0) for c in net.graph.get(b, {})]
    self.lane_index = min(candidates, key=lambda li: net.get_lane(li).distance_with_heading(self.position, self.heading))
    self.lane = net.get_lane(self.lane_index)
    if road.record_history:
        self.history.appendleft(self.create_from(self))


Vehicle.on_state_update = _fast_on_state_update

ROWS, COLS = 2, 3
ACCESS = 50.0                                   # road length on each side of a junction [m]
LANE_W = AbstractLane.DEFAULT_WIDTH             # 4 m
OUTER = LANE_W + 5 + LANE_W / 2                 # 11 m: from junction centre to where the access road starts
SPACING = 2 * (ACCESS + OUTER)                  # 122 m: junction centres, so neighbouring roads meet exactly
SEE_RADIUS = 90.0                               # cars within this distance of the junction are shown to the agent
TURN_NAMES = {1: "LEFT", 2: "STRAIGHT", 3: "RIGHT"}
# arm c of a junction: 0 = south (+y), 1 = west (-x), 2 = north (-y), 3 = east (+x)
NEIGHBOUR = {0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}   # arm -> (d_row, d_col)


def junction_id(r, c):
    return r * COLS + c


def center(k):
    r, c = divmod(k, COLS)
    return np.array([c * SPACING, r * SPACING], dtype=float)


def outer_node(k, arm):
    """Shared node name between two neighbouring junctions, or a map-edge node."""
    r, c = divmod(k, COLS)
    dr, dc = NEIGHBOUR[arm]
    if 0 <= r + dr < ROWS and 0 <= c + dc < COLS:
        a, b = sorted([k, junction_id(r + dr, c + dc)])
        return f"m{a}_{b}"
    return f"edge{k}_{arm}"


class JourneyEnv(IntersectionEnv):
    _options = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "duration": 90,
            "initial_vehicle_count": 16,
            "spawn_probability": 0.6,
            "spawn_attempts": 2,          # spawn tries per second at random map edges
            "max_vehicles": 26,
            "min_junctions": 3,
            "keep_previous_junction": True,   # blind-spot fix (see current_junction)
        })
        return config

    # ------------------------------------------------------------------ map
    def _make_road(self) -> None:
        """The original intersection (highway_env/envs/intersection_env.py), copied 6 times with shared roads."""
        net = JourneyNetwork()
        n, cl, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        right_r = LANE_W + 5
        left_r = right_r + LANE_W
        for k in range(ROWS * COLS):
            o = center(k)
            for corner in range(4):
                angle = np.radians(90 * corner)
                pr = 3 if corner % 2 else 1
                rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                ir, il = f"ir{corner}:{k}", lambda c: f"il{c % 4}:{k}"
                net.add_lane(outer_node(k, corner), ir, StraightLane(
                    o + rot @ [LANE_W / 2, ACCESS + OUTER], o + rot @ [LANE_W / 2, OUTER],
                    line_types=[s, cl], priority=pr, speed_limit=10.0))
                net.add_lane(ir, il(corner - 1), CircularLane(
                    o + rot @ [OUTER, OUTER], right_r, angle + np.radians(180), angle + np.radians(270),
                    line_types=[n, cl], priority=pr, speed_limit=10.0))
                net.add_lane(ir, il(corner + 1), CircularLane(
                    o + rot @ [-left_r + LANE_W / 2, left_r - LANE_W / 2], left_r, angle, angle + np.radians(-90),
                    clockwise=False, line_types=[n, n], priority=pr - 1, speed_limit=10.0))
                net.add_lane(ir, il(corner + 2), StraightLane(
                    o + rot @ [LANE_W / 2, OUTER], o + rot @ [LANE_W / 2, -OUTER],
                    line_types=[s, n], priority=pr, speed_limit=10.0))
                net.add_lane(il(corner - 1), outer_node(k, (corner - 1) % 4), StraightLane(
                    o + rot @ np.flip([LANE_W / 2, OUTER]), o + rot @ np.flip([LANE_W / 2, ACCESS + OUTER]),
                    line_types=[n, cl], priority=pr, speed_limit=10.0))
        self.road = RegulatedRoad(network=net, np_random=self.np_random,
                                  record_history=self.config["show_trajectories"],
                                  neighbour_vehicles_connected_lanes=self.config["neighbour_vehicles_connected_lanes"])
        self.edge_entries = [(k, a) for k in range(ROWS * COLS) for a in range(4)
                             if outer_node(k, a).startswith("edge")]

    def entry_lane(self, k, arm):
        return (outer_node(k, arm), f"ir{arm}:{k}", 0)

    def plan(self, start_lane, goal_node):
        path = self.road.network.shortest_path(start_lane[1], goal_node)
        return [start_lane] + [(path[i], path[i + 1], 0) for i in range(len(path) - 1)] if path else []

    @staticmethod
    def junctions_of(route):
        """[(junction, arm in, arm out, turn)] for every junction crossed by a route (list of lane indices)."""
        out = []
        for a, b, _ in route:
            if a.startswith("ir") and b.startswith("il"):
                k = int(a.split(":")[1])
                cin, cout = int(a[2]), int(b[2])
                # entering on arm cin and leaving on arm cout: right turn -> cout = cin-1, left -> cin+1
                out.append((k, cin, cout, {3: 3, 1: 1, 2: 2}[(cout - cin) % 4]))
        return out

    # ------------------------------------------------------------------ traffic
    def _make_vehicles(self, n_vehicles: int = 16) -> None:
        vt = class_from_path(self.config["other_vehicles_type"])
        vt.DISTANCE_WANTED, vt.COMFORT_ACC_MAX, vt.COMFORT_ACC_MIN = 7, 6, -3

        # the journey: random map-edge entry -> random map-edge exit with at least `min_junctions` junctions
        if JourneyEnv._options is None:  # the map never changes: list the possible journeys once
            JourneyEnv._options = []
            for k0, a0 in self.edge_entries:
                for k1, a1 in self.edge_entries:
                    js = self.junctions_of(self.plan(self.entry_lane(k0, a0), outer_node(k1, a1)))
                    if (k0, a0) != (k1, a1) and len(js) >= self.config["min_junctions"] and any(t != 2 for *_, t in js):
                        JourneyEnv._options.append((k0, a0, outer_node(k1, a1)))
        k0, a0, goal = JourneyEnv._options[self.np_random.integers(len(JourneyEnv._options))]

        # traffic everywhere first, then a few seconds of simulation so it looks natural
        all_entries = [(k, a) for k in range(ROWS * COLS) for a in range(4)]
        for _ in range(n_vehicles * 3):
            if len(self.road.vehicles) >= n_vehicles:
                break
            k, a = all_entries[self.np_random.integers(len(all_entries))]
            self._add_car(self.entry_lane(k, a), self.np_random.uniform(0, ACCESS - 5))
        for _ in range(2 * self.config["simulation_frequency"]):
            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])

        lane = self.road.network.get_lane(self.entry_lane(k0, a0))
        ego = self.action_type.vehicle_class(self.road, lane.position(12.0 + self.np_random.normal(), 0),
                                             speed=lane.speed_limit, heading=lane.heading_at(12.0))
        ego.plan_route_to(goal)
        ego.speed_index = ego.speed_to_index(lane.speed_limit)
        ego.target_speed = ego.index_to_speed(ego.speed_index)
        self.road.vehicles = [v for v in self.road.vehicles if np.linalg.norm(v.position - ego.position) > 20]
        self.road.vehicles.append(ego)
        self.controlled_vehicles = [ego]
        self.goal_node = goal
        self.final_lane = ego.route[-1][:2]
        self.journey = self.junctions_of(ego.route)      # [(junction, arm_in, arm_out, turn)]
        self.journey_length = sum(self.road.network.get_lane(l).length for l in ego.route)

    def _add_car(self, lane_index, longitudinal, speed=None):
        vt = class_from_path(self.config["other_vehicles_type"])
        car = vt.make_on_lane(self.road, lane_index, longitudinal=longitudinal,
                              speed=speed if speed is not None else 8.0 + self.np_random.normal())
        if any(np.linalg.norm(v.position - car.position) < 15 for v in self.road.vehicles):
            return None
        k1, a1 = self.edge_entries[self.np_random.integers(len(self.edge_entries))]
        car.plan_route_to(outer_node(k1, a1))
        if len(car.route) < 2:
            return None
        car.randomize_behavior()
        self.road.vehicles.append(car)
        return car

    def _spawn_vehicle(self, longitudinal: float = 0, position_deviation: float = 1.0, speed_deviation: float = 1.0,
                       spawn_probability: float = 0.6, go_straight: bool = False) -> None:
        for _ in range(self.config["spawn_attempts"]):
            if len(self.road.vehicles) >= self.config["max_vehicles"] or self.np_random.uniform() > spawn_probability:
                continue
            k, a = self.edge_entries[self.np_random.integers(len(self.edge_entries))]
            self._add_car(self.entry_lane(k, a), self.np_random.uniform(0, 5))

    def _clear_vehicles(self) -> None:
        """
        Remove cars that left the map, plus two fixes for long episodes (called once per second):
        - tow truck: other cars that crashed into each other are removed after 2 s (highway-env leaves them
          frozen forever, which blocks the junction for the rest of a 90 s journey)
        - gridlock breaker: a car standing still for 15 s (deadlocked yielding) is removed
        """
        def leaving(v):
            return (v.lane_index[1].startswith("edge")
                    and v.lane.local_coordinates(v.position)[0] >= v.lane.length - 4 * v.LENGTH)

        keep = []
        for v in self.road.vehicles:
            if v in self.controlled_vehicles:
                keep.append(v)
                continue
            v.crash_age = getattr(v, "crash_age", 0) + 1 if v.crashed else 0
            v.stuck_age = getattr(v, "stuck_age", 0) + 1 if v.speed < 0.5 else 0
            if leaving(v) or v.route is None or v.crash_age > 2 or v.stuck_age > 15:
                continue
            keep.append(v)
        self.road.vehicles = keep

    def has_arrived(self, vehicle, exit_distance: float = 25) -> bool:
        return (vehicle is self.vehicle and vehicle.lane_index[:2] == self.final_lane
                and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance)

    # ------------------------------------------------------------------ what the agent sees
    def current_junction(self):
        """Index into self.journey of the junction we are approaching / crossing (last one after it)."""
        ego = self.vehicle
        a, b, _ = ego.lane_index
        for i, (k, cin, cout, _) in enumerate(self.journey):
            if a == f"ir{cin}:{k}" or (a.startswith("o") or a.startswith("m") or a.startswith("edge")) and b == f"ir{cin}:{k}":
                return i
            if a == f"il{cout}:{k}":
                # On the exit road we keep watching the junction we just crossed until halfway to the next one:
                # switching earlier would make the cars still crossing behind us vanish from view (a blind spot
                # that caused rear/side crashes right after each junction).
                if self.config["keep_previous_junction"] and                         ego.lane.local_coordinates(ego.position)[0] < ego.lane.length / 2:
                    return i
                return min(i + 1, len(self.journey) - 1)
        return len(self.journey) - 1

    @staticmethod
    def next_exit_point(vehicle, road):
        """End of the exit road of the next junction on this car's route (its 'turn signal' target)."""
        for a, b, _ in vehicle.route or []:
            if a.startswith("il"):
                lane = road.network.get_lane((a, b, 0))
                return lane.position(lane.length, 0)
        return vehicle.destination

    def local_obs(self):
        i = self.current_junction()
        k, cin, cout, turn = self.journey[i]
        c, ego = center(k), self.vehicle
        lane = self.road.network.get_lane((f"il{cout}:{k}", outer_node(k, cout), 0))
        self.target = lane.position(lane.length, 0)    # the exit the agent must take at this junction
        self.junction_index, self.junction = i, k

        def row(v, goal):
            d = goal - v.position
            d = d / (np.linalg.norm(d) + 1e-9)
            vel = v.velocity
            return [1.0, *np.clip((v.position - c) / 100, -1, 1), *np.clip(vel / 20, -1, 1),
                    np.cos(v.heading), np.sin(v.heading), *d]

        rows = [row(ego, self.target)]
        others = [v for v in self.road.vehicles if v is not ego and np.linalg.norm(v.position - c) < SEE_RADIUS]
        others.sort(key=lambda v: np.linalg.norm(v.position - ego.position))
        others = others[:14]
        self.observed = others
        rows += [row(v, self.next_exit_point(v, self.road)) for v in others]
        obs = np.zeros((15, 9), dtype=np.float32)
        obs[:len(rows)] = rows
        self.obs_rng.shuffle(obs[1:len(rows)])  # own generator: what the agent sees never changes the traffic
        return obs

    def reset(self, *, seed=None, options=None):
        self.obs_rng = np.random.default_rng(seed)
        _, info = super().reset(seed=seed, options=options)
        return self.local_obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        return self.local_obs(), reward, terminated, truncated, info

    @property
    def route_name(self):
        return f"{len(self.journey)} junctions: " + "-".join(TURN_NAMES[t] for *_, t in self.journey)

    def progress(self):
        """Fraction of the journey driven (route[0] is the lane we are on, the rest is still ahead)."""
        ego, remaining = self.vehicle, 0.0
        for idx, (a, b, _) in enumerate(ego.route):
            lane = self.road.network.get_lane((a, b, 0))
            remaining += lane.length - (lane.local_coordinates(ego.position)[0] if idx == 0 else 0)
        return float(np.clip(1 - remaining / self.journey_length, 0, 1))


gym.register(id="intersection-journey-v0", entry_point=JourneyEnv)


def make_journey_env(render_mode=None, extra_config=None):
    from common import MULTI_OBS
    config = {"observation": MULTI_OBS}
    config.update(extra_config or {})
    return gym.make("intersection-journey-v0", render_mode=render_mode, config=config)
