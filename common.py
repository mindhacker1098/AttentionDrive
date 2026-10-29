"""
Shared pieces for the Social Attention project (Leurent & Mercat, NeurIPS 2019 ML4AD workshop).

- make_env:        task "left"  = the paper's task: a car must turn left at a busy 4-way intersection
                                  (highway-env "intersection-v0", actions SLOWER / IDLE / FASTER)
                   task "multi" = OUR EXTENSION: start on any of the 4 roads, go left / straight / right
                                  (12 routes, one agent), goal + other cars' turn signals in the state
                                  (the paper's "goal conditioning" future-work idea, Section 4)
- FCNList:         baseline 1, "FCN/List"    (paper Table 1: layers [128, 128], ~30k parameters)
- GridCNN:         baseline 2, "CNN/Grid"    (32x32 grid of 2 m cells, 3 conv layers + [20] head, ~31k)
- EgoAttentionNet: the paper's model         (encoders [64, 64], 2 heads, decoder [64, 64], ~30k)
- ReplayBuffer, epsilon schedule, CSV logger

All settings are copied from the authors' own configs in github.com/eleurent/rl-agents
(scripts/configs/IntersectionEnv).
"""
import csv
import os
import warnings

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", category=DeprecationWarning)  # "intersection-v0 is out of date" (v0 = the paper's task)

import gymnasium as gym
import highway_env  # noqa: F401  (registers the environments)

gym.logger.min_level = 40  # hide gymnasium's "out of date" notice (v0 is exactly the paper's task)
import numpy as np
import torch
import torch.nn as nn
from highway_env.envs.intersection_env import IntersectionEnv
from highway_env.utils import class_from_path

MODELS = ["fcn", "cnn", "attention"]
LABELS = {"fcn": "FCN/List", "cnn": "CNN/Grid", "attention": "Ego-Attention", "random": "Random"}
ACTIONS = ["SLOWER", "IDLE", "FASTER"]

FEATURES = ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
FEATURES_RANGE = {"x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]}

# "List of features": one row per vehicle (row 0 = our car), up to 15 vehicles, 7 numbers each.
# Rows of the other cars are SHUFFLED every step, so the network cannot rely on their order.
LIST_OBS = {
    "type": "Kinematics",
    "vehicles_count": 15,
    "features": FEATURES,
    "features_range": FEATURES_RANGE,
    "absolute": True,
    "order": "shuffled",
}

# "Spatial grid": a 32x32 bird's-eye picture around our car, each cell 2 m x 2 m, 7 channels.
GRID_OBS = {
    "type": "OccupancyGrid",
    "vehicles_count": 15,
    "features": FEATURES,
    "features_range": FEATURES_RANGE,
    "grid_size": [[-32, 32], [-32, 32]],
    "grid_step": [2, 2],
    "absolute": False,
}

# Our own experiment: heavier traffic than the agents were trained on (default: 10 cars, 60% spawn chance).
DENSE_TRAFFIC = {"initial_vehicle_count": 15, "spawn_probability": 0.9}

# Multi-route extension: 2 extra features per car = direction to its destination (cos_d, sin_d).
# For our car this is the GOAL; for the other cars it works like a turn signal.
MULTI_FEATURES = FEATURES + ["cos_d", "sin_d"]
MULTI_OBS = dict(LIST_OBS, features=MULTI_FEATURES, observe_intentions=True)

TASKS = ["left", "multi", "journey"]
TURNS = {1: "LEFT", 2: "STRAIGHT", 3: "RIGHT"}
ROADS = {0: "south", 1: "west", 2: "north", 3: "east"}  # west/east road = priority road


class MultiRouteIntersectionEnv(IntersectionEnv):
    """
    Same intersection, but every episode our car starts on a random road (4) and gets a random
    destination: left, straight or right (3) -> 12 routes. West/east is the priority road, so some
    routes must give way and others have right of way.
    """

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        # same traffic set-up as the original (highway_env/envs/intersection_env.py), only the ego route changes
        vehicle_type = class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(3):
            for _ in range(self.config["simulation_frequency"]):
                self.road.act()
                self.road.step(1 / self.config["simulation_frequency"])
        self._spawn_vehicle(60, spawn_probability=1.0, go_straight=True,
                            position_deviation=0.1, speed_deviation=0.0)

        self.start_road = int(self.np_random.integers(4))
        self.turn = int(self.np_random.integers(1, 4))
        ego_lane = self.road.network.get_lane((f"o{self.start_road}", f"ir{self.start_road}", 0))
        ego = self.action_type.vehicle_class(
            self.road, ego_lane.position(60.0 + 5.0 * self.np_random.normal(1.0), 0.0),
            speed=ego_lane.speed_limit, heading=ego_lane.heading_at(60.0))
        ego.plan_route_to(f"o{(self.start_road + self.turn) % 4}")
        ego.speed_index = ego.speed_to_index(ego_lane.speed_limit)
        ego.target_speed = ego.index_to_speed(ego.speed_index)
        self.road.vehicles.append(ego)
        self.controlled_vehicles = [ego]
        for v in self.road.vehicles.copy():  # prevent collisions at the very start
            if v is not ego and np.linalg.norm(v.position - ego.position) < 20:
                self.road.vehicles.remove(v)

    @property
    def route_name(self):
        return f"{ROADS[self.start_road]} -> {TURNS[self.turn]}"


gym.register(id="intersection-multiroute-v0", entry_point=MultiRouteIntersectionEnv)


def n_features(task):
    return len(MULTI_FEATURES) if task in ("multi", "journey") else len(FEATURES)


def make_env(model, seed=None, render_mode=None, dense=False, task="left", extra_config=None):
    """Intersection task. Reward: +1 at full speed, -5 for a crash, +1 for arriving. Episode = 13 s, 1 decision/s."""
    if task == "journey":  # long journey through a 6-junction city (journey.py)
        from journey import make_journey_env
        extra = dict(extra_config or {})
        if dense:
            extra.update({"max_vehicles": 34, "spawn_probability": 0.9})
        return make_journey_env(render_mode=render_mode, extra_config=extra)
    if task == "multi":
        config, env_id = {"observation": MULTI_OBS, "duration": 15}, "intersection-multiroute-v0"
    else:
        config, env_id = {"observation": GRID_OBS if model == "cnn" else LIST_OBS}, "intersection-v0"
    if dense:
        config.update(DENSE_TRAFFIC)
    config.update(extra_config or {})
    env = gym.make(env_id, render_mode=render_mode, config=config)
    if seed is not None:
        env.action_space.seed(seed)
    return env


# ----------------------------------------------------------------------------- networks

def mlp(sizes, out=None):
    """Linear+ReLU for every hidden size, then an optional plain Linear output (same as rl-agents)."""
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    if out is not None:
        layers.append(nn.Linear(sizes[-1], out))
    return nn.Sequential(*layers)


class FCNList(nn.Module):
    """Flatten the 15x7 table into 105 numbers and use a plain fully-connected network."""

    def __init__(self, n_actions=3, n_vehicles=15, n_features=7):
        super().__init__()
        self.net = mlp([n_vehicles * n_features, 128, 128], out=n_actions)

    def forward(self, x):
        return self.net(x.flatten(1))


class GridCNN(nn.Module):
    """Small CNN on the 7x32x32 occupancy grid: 32 -> 16 -> 8 -> 4, then a 20-unit head."""

    def __init__(self, n_actions=3, n_features=7, size=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_features, 16, 2, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 2, stride=2), nn.ReLU(),
        )
        self.head = mlp([64 * (size // 8) ** 2, 20], out=n_actions)

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))


class EgoAttentionNet(nn.Module):
    """
    The paper's model (Figure 2 and 3).

    1. Encode every vehicle with a shared small MLP (one for our car, one for the others).
    2. Our car asks a QUERY ("who is dangerous for me?"); every vehicle answers with a KEY.
       attention = softmax(query . key / sqrt(d)) -> one weight per vehicle, summing to 1.
    3. Mix the vehicles' VALUES with these weights, add our own encoding back (residual).
    4. Decode into 3 Q-values (SLOWER, IDLE, FASTER).

    Order of the other cars does not matter (permutation invariant), and empty rows are masked out,
    so any number of cars works.
    """

    def __init__(self, n_actions=3, n_features=7, d=64, heads=2):
        super().__init__()
        self.d, self.heads = d, heads
        self.ego_encoder = mlp([n_features, 64, d])
        self.others_encoder = mlp([n_features, 64, d])
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.combine = nn.Linear(d, d, bias=False)
        self.decoder = mlp([d, 64, 64], out=n_actions)

    def attend(self, x):
        """x: (batch, vehicles, features), row 0 = our car. Returns features (B, d) and attention (B, heads, vehicles)."""
        B, N, _ = x.shape
        h, dh = self.heads, self.d // self.heads
        ego = self.ego_encoder(x[:, :1])                                # (B, 1, d)
        everyone = torch.cat([ego, self.others_encoder(x[:, 1:])], 1)  # (B, N, d)

        q = self.query(ego).view(B, 1, h, dh).transpose(1, 2)           # (B, h, 1, dh)
        k = self.key(everyone).view(B, N, h, dh).transpose(1, 2)        # (B, h, N, dh)
        v = self.value(everyone).view(B, N, h, dh).transpose(1, 2)      # (B, h, N, dh)

        scores = q @ k.transpose(-2, -1) / dh ** 0.5                    # (B, h, 1, N)
        absent = (x[:, :, 0] < 0.5).view(B, 1, 1, N)                    # presence feature = 0 -> empty row
        attention = scores.masked_fill(absent, -1e9).softmax(-1)

        mixed = (attention @ v).reshape(B, self.d)                      # heads side by side
        out = (self.combine(mixed) + ego.squeeze(1)) / 2                # residual, as in rl-agents
        return out, attention.squeeze(2)

    def forward(self, x):
        return self.decoder(self.attend(x)[0])


def build_model(model, task="left"):
    return {"fcn": FCNList, "cnn": GridCNN, "attention": EgoAttentionNet}[model](n_features=n_features(task))


def count_parameters(net):
    return sum(p.numel() for p in net.parameters())


# ----------------------------------------------------------------------------- DQN helpers

def epsilon_at(step, tau=15000, final=0.05, start=1.0):
    """Exploration rate: starts at 1 (fully random) and decays smoothly towards 0.05."""
    return final + (start - final) * np.exp(-step / tau)


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, seed=0):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminal):
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminals[i] = next_obs, terminal
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = self.rng.integers(0, self.size, batch_size)
        t = lambda a: torch.as_tensor(a[idx], device=device)
        return t(self.obs), t(self.actions), t(self.rewards), t(self.next_obs), t(self.terminals)


class CSVLogger:
    def __init__(self, path, fields):
        new = not os.path.exists(path)
        self.file = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        if new:
            self.writer.writeheader()

    def log(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def episode_outcome(env, terminated, crashed):
    """'arrived' (made the left turn), 'crashed', or 'timeout' (13 s ran out, usually waiting too long)."""
    if crashed:
        return "crashed"
    if terminated and env.unwrapped.has_arrived(env.unwrapped.vehicle):
        return "arrived"
    return "timeout"
