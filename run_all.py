"""
Train all three agents x 3 seeds in parallel (9 runs, ~1-1.5 h on an 8-core laptop), then plot and evaluate.

    python run_all.py              # everything
    python run_all.py --jobs 6     # fewer runs at once if the laptop gets hot
    python run_all.py --task multi --models attention fcn --seeds 1 2 --episodes 6000   # 12-route extension
Finished runs (runs/<name>/model.pt exists) are skipped, so it is safe to stop and restart.
"""
import argparse
import os
import subprocess
import sys
import time

from common import MODELS, TASKS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default="left")
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--jobs", type=int, default=9)
    p.add_argument("--episodes", type=int, default=4000)
    args = p.parse_args()

    os.makedirs("logs", exist_ok=True)
    prefix = "" if args.task == "left" else f"{args.task}_"
    queue = [(m, s) for s in args.seeds for m in args.models
             if not os.path.exists(os.path.join("runs", f"{prefix}{m}_s{s}", "model.pt"))]
    running, start = [], time.time()
    print(f"{len(queue)} runs to do, {args.jobs} at a time", flush=True)
    while queue or running:
        while queue and len(running) < args.jobs:
            m, s = queue.pop(0)
            log = open(os.path.join("logs", f"{prefix}{m}_s{s}.log"), "w")
            cmd = [sys.executable, "train.py", "--model", m, "--seed", str(s), "--episodes", str(args.episodes),
                   "--task", args.task]
            running.append((m, s, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
            print(f"[start] {m} seed {s}", flush=True)
        for item in running[:]:
            if item[2].poll() is not None:
                running.remove(item)
                print(f"[done ] {item[0]} seed {item[1]} (exit {item[2].returncode}, "
                      f"{(time.time() - start) / 60:.0f} min)", flush=True)
        time.sleep(10)

    subprocess.run([sys.executable, "evaluate.py", "--all"])
    subprocess.run([sys.executable, "evaluate.py", "--all", "--dense"])
    subprocess.run([sys.executable, "plot.py"])


if __name__ == "__main__":
    main()
