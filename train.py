"""
Train one DQN agent on the intersection task.

    python train.py --model attention --seed 1
    python train.py --model fcn --seed 1
    python train.py --model cnn --seed 1
    python train.py --model attention --task multi --seed 1 --episodes 6000   # our 12-route extension
    python train.py --model attention --task journey --init runs/multi_attention_s1/model.pt --episodes 600 \
                    --eps-start 0.2 --eps-tau 5000                             # fine-tune on the long journey

Same DQN settings for all three networks (from the authors' rl-agents configs):
Double DQN, gamma 0.95, Adam lr 5e-4, batch 64, replay 15k, target copy every 512 steps,
epsilon 1 -> 0.05 (tau 15k steps), gradients clipped to [-1, 1], 4000 episodes.

Output in runs/<model>_s<seed>/ (or runs/multi_<model>_s<seed>/): progress.csv (one row per episode), model.pt, config.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (MODELS, TASKS, CSVLogger, ReplayBuffer, build_model, count_parameters,
                    episode_outcome, epsilon_at, make_env)

FIELDS = ["episode", "step", "time_s", "return", "length", "speed", "outcome", "epsilon", "route"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--task", choices=TASKS, default="left", help="left = paper's task, multi = 12 routes, journey = 6-junction city")
    p.add_argument("--init", default=None, help="start from these weights (fine-tuning)")
    p.add_argument("--eps-start", type=float, default=1.0, help="initial exploration (lower when fine-tuning)")
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer", type=int, default=15000)
    p.add_argument("--target-update", type=int, default=512)
    p.add_argument("--eps-tau", type=float, default=15000)
    p.add_argument("--run-dir", default=None)
    args = p.parse_args()

    prefix = "" if args.task == "left" else f"{args.task}_"
    run_dir = args.run_dir or os.path.join("runs", f"{prefix}{args.model}_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "model.pt")):
        print(f"{run_dir} already finished, skipping")
        return
    progress = os.path.join(run_dir, "progress.csv")
    if os.path.exists(progress):
        os.remove(progress)  # unfinished earlier attempt: start clean

    torch.set_num_threads(1)  # small networks: several runs in parallel use the CPU better
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cpu"

    env = make_env(args.model, seed=args.seed, task=args.task)
    obs, _ = env.reset(seed=args.seed)
    q_net = build_model(args.model, args.task).to(device)
    target_net = build_model(args.model, args.task).to(device)
    if args.init:
        q_net.load_state_dict(torch.load(args.init, map_location=device))
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer, obs.shape, seed=args.seed)

    config = dict(vars(args), parameters=count_parameters(q_net), obs_shape=list(obs.shape))
    json.dump(config, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    print(f"{args.model} seed {args.seed}: {config['parameters']} parameters, obs {obs.shape}")

    logger = CSVLogger(progress, FIELDS)
    step, start = 0, time.time()
    for episode in range(1, args.episodes + 1):
        ep_return, speeds, done = 0.0, [], False
        route = getattr(env.unwrapped, "route_name", "south -> LEFT")
        while not done:
            eps = epsilon_at(step, args.eps_tau, start=args.eps_start)
            if rng.random() < eps:
                action = int(rng.integers(env.action_space.n))
            else:
                with torch.no_grad():
                    action = int(q_net(torch.as_tensor(obs[None], device=device)).argmax(1))

            next_obs, reward, terminated, truncated, info = env.step(action)
            # crash / arrival = true end (no future value); running out of time = cut short (still bootstrap)
            buffer.add(obs, action, reward, next_obs, float(terminated))
            obs, done = next_obs, terminated or truncated
            ep_return += reward
            speeds.append(info["speed"])
            step += 1

            if buffer.size >= args.batch_size:
                s, a, r, s2, term = buffer.sample(args.batch_size, device)
                q = q_net(s).gather(1, a[:, None]).squeeze(1)
                with torch.no_grad():  # Double DQN: online net picks the action, target net scores it
                    best = q_net(s2).argmax(1, keepdim=True)
                    target = r + args.gamma * (1 - term) * target_net(s2).gather(1, best).squeeze(1)
                loss = F.mse_loss(q, target)
                optimizer.zero_grad()
                loss.backward()
                for prm in q_net.parameters():
                    prm.grad.data.clamp_(-1, 1)
                optimizer.step()
            if step % args.target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

        outcome = episode_outcome(env, terminated, info["crashed"])
        logger.log({"episode": episode, "step": step, "time_s": round(time.time() - start, 1),
                    "return": round(ep_return, 3), "length": len(speeds),
                    "speed": round(float(np.mean(speeds)), 3), "outcome": outcome, "epsilon": round(eps, 4), "route": route})
        if episode % 100 == 0:
            torch.save(q_net.state_dict(), os.path.join(run_dir, "checkpoint.pt"))
            print(f"[{prefix}{args.model} s{args.seed}] ep {episode}/{args.episodes} step {step} "
                  f"eps {eps:.2f} return {ep_return:.2f} {outcome} ({(time.time() - start) / 60:.1f} min)", flush=True)
        obs, _ = env.reset()

    torch.save(q_net.state_dict(), os.path.join(run_dir, "model.pt"))
    logger.close()
    env.close()
    print(f"done in {(time.time() - start) / 60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
