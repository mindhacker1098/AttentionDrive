"""
Test agents on the LONG JOURNEY: the same 40 random city journeys (3-4 junctions, several turns) for every agent.

    python evaluate_journey.py                       # every multi_/journey_ run that has a model.pt + random driver
    python evaluate_journey.py --episodes 20 --dense

Agents trained on the 12-route task never saw a multi-junction map ("zero-shot"); journey_ runs were fine-tuned on it.
Result: results/journey_eval_<normal|dense>.csv  (finish / crash / timeout %, junctions passed, progress, time)
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from common import build_model, episode_outcome, make_env

TEST_SEED = 20_000


def run(job):
    run_dir, episodes, dense, keep_prev = job
    torch.set_num_threads(1)
    net, name = None, "random"
    if run_dir != "random":
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        net = build_model(cfg["model"], cfg.get("task", "multi"))
        net.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"))
        net.eval()
        name = os.path.basename(run_dir)
    env = make_env("attention", task="journey", dense=dense, extra_config={"keep_previous_junction": keep_prev})
    u, rng, rows = env.unwrapped, np.random.default_rng(0), []
    for i in range(episodes):
        obs, _ = env.reset(seed=TEST_SEED + i)
        done = False
        while not done:
            if net is None:
                action = int(rng.integers(3))
            else:
                with torch.no_grad():
                    action = int(net(torch.as_tensor(obs[None])).argmax(1))
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        outcome = episode_outcome(env, terminated, info["crashed"])
        rows.append({"outcome": outcome, "junctions": len(u.journey),
                     "passed": len(u.journey) if outcome == "arrived" else u.junction_index,
                     "progress": 1.0 if outcome == "arrived" else u.progress(), "time": u.time})
    env.close()
    df = pd.DataFrame(rows)
    fin = df[df.outcome == "arrived"]
    return {"agent": name + ("" if keep_prev else " (no blind-spot fix)"),
            "finish_%": 100 * (df.outcome == "arrived").mean(), "crash_%": 100 * (df.outcome == "crashed").mean(),
            "timeout_%": 100 * (df.outcome == "timeout").mean(),
            "junctions_passed": df.passed.mean(), "junctions_total": df.junctions.mean(),
            "progress_%": 100 * df.progress.mean(), "finish_time_s": fin.time.mean() if len(fin) else np.nan}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--dense", action="store_true")
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--compare-fix", action="store_true", help="also test without the blind-spot fix")
    args = p.parse_args()
    runs = args.runs or sorted(os.path.dirname(f) for pat in ("runs/multi_*/model.pt", "runs/journey_*/model.pt")
                               for f in glob.glob(pat))
    jobs = [(r, args.episodes, args.dense, True) for r in runs + ["random"]]
    if args.compare_fix:
        jobs += [(r, args.episodes, args.dense, False) for r in runs]
    with Pool(len(jobs)) as pool:
        df = pd.DataFrame(pool.map(run, jobs))
    traffic = "dense" if args.dense else "normal"
    print(f"\n=== long journey, {traffic} traffic, {args.episodes} journeys per agent ===")
    print(df.round(1).to_string(index=False))
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/journey_eval_{traffic}.csv", index=False)
    print(f"-> results/journey_eval_{traffic}.csv")


if __name__ == "__main__":
    main()
