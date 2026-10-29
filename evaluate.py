"""
Test trained agents with no exploration on fixed traffic scenarios (the same ones for every agent).

    python evaluate.py --all                 # every finished run + a random driver
    python evaluate.py --all --dense         # our experiment: heavier traffic than in training
    python evaluate.py --run runs/attention_s1

Results go to results/eval_<normal|dense>.csv (paper task, 100 scenarios) and
results/multi_eval_<normal|dense>.csv (12-route extension, 150 scenarios, with success per turn type).
Outcome of each episode: arrived (made it through), crashed, or timeout (time ran out).
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from common import LABELS, TURNS, build_model, episode_outcome, make_env

TEST_SEED = 10_000  # test scenarios never used in training (training uses seeds 1-3)
EPISODES = {"left": 100, "multi": 150}


def load_agent(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    task = cfg.get("task", "left")
    net = build_model(cfg["model"], task)
    net.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"))
    net.eval()
    return cfg["model"], cfg["seed"], task, net


def evaluate(job):
    run_dir, task, episodes, dense = job
    torch.set_num_threads(1)
    if run_dir == "random":
        model, seed, net = "random", 0, None
    else:
        model, seed, task, net = load_agent(run_dir)
    env = make_env("fcn" if net is None else model, dense=dense, task=task)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(episodes):
        obs, _ = env.reset(seed=TEST_SEED + i)
        turn = TURNS[getattr(env.unwrapped, "turn", 1)]
        done, ret, speeds = False, 0.0, []
        while not done:
            if net is None:
                action = int(rng.integers(3))
            else:
                with torch.no_grad():
                    action = int(net(torch.as_tensor(obs[None])).argmax(1))
            obs, r, terminated, truncated, info = env.step(action)
            done, ret = terminated or truncated, ret + r
            speeds.append(info["speed"])
        rows.append({"outcome": episode_outcome(env, terminated, info["crashed"]), "turn": turn,
                     "return": ret, "speed": np.mean(speeds)})
    env.close()
    df = pd.DataFrame(rows)
    result = {"task": task, "model": model, "seed": seed,
              "success_%": 100 * (df.outcome == "arrived").mean(),
              "crash_%": 100 * (df.outcome == "crashed").mean(),
              "timeout_%": 100 * (df.outcome == "timeout").mean(),
              "return": df["return"].mean(), "speed": df.speed.mean()}
    if task == "multi":
        for t in TURNS.values():
            sub = df[df.turn == t]
            result[f"success_{t}_%"] = 100 * (sub.outcome == "arrived").mean() if len(sub) else np.nan
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="one run directory")
    p.add_argument("--all", action="store_true", help="every finished run in runs/ + a random driver")
    p.add_argument("--episodes", type=int, default=None, help="default: 100 (paper task), 150 (multi-route)")
    p.add_argument("--dense", action="store_true", help="heavier traffic than in training")
    args = p.parse_args()

    if args.all:
        runs = sorted(os.path.dirname(f) for f in glob.glob("runs/*/model.pt"))
        tasks = {}
        for r in runs:
            task = json.load(open(os.path.join(r, "config.json"))).get("task", "left")
            if task in EPISODES:  # journey runs are tested by evaluate_journey.py
                tasks.setdefault(task, []).append(r)
        jobs = [(r, task, args.episodes or EPISODES[task], args.dense)
                for task, rs in tasks.items() for r in rs + ["random"]]
    else:
        task = json.load(open(os.path.join(args.run, "config.json"))).get("task", "left")
        jobs = [(args.run, task, args.episodes or EPISODES[task], args.dense)]
    with Pool(min(len(jobs), os.cpu_count() or 4)) as pool:
        results = pd.DataFrame(pool.map(evaluate, jobs))

    traffic = "dense" if args.dense else "normal"
    for task, df in results.groupby("task"):
        df = df.dropna(axis=1, how="all")
        print(f"\n=== task '{task}', {traffic} traffic, {jobs[0][2] if not args.all else EPISODES[task]} "
              f"test episodes per agent ===")
        print(df.drop(columns="task").round(1).to_string(index=False))
        summary = df.drop(columns=["task", "seed"]).groupby("model").agg(["mean", "std"]).round(1)
        summary.index = [LABELS[m] for m in summary.index]
        print("\nmean (std over seeds):")
        print(summary.to_string())
        if args.all:
            os.makedirs("results", exist_ok=True)
            out = f"results/{'' if task == 'left' else task + '_'}eval_{traffic}.csv"
            df.to_csv(out, index=False)
            print(f"-> {out}")


if __name__ == "__main__":
    main()
