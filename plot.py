"""
Make the slide figures.

    python plot.py

plots/training_curves.png  - our version of the paper's Figure 4 (total reward, episode length, speed)
                             + success rate, averaged over seeds with 95% confidence bands
plots/eval_normal.png      - test results on 100 fixed scenarios (needs evaluate.py --all)
plots/eval_dense.png       - same with heavier traffic (needs evaluate.py --all --dense)
plots/multi_*.png          - the same figures for our 12-route extension, + success per turn type
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import LABELS, MODELS

# Validated categorical palette (slots 1-3), light surface, recessive ink.
COLORS = {"attention": "#2a78d6", "fcn": "#eb6834", "cnn": "#1baf7a", "random": "#8a8984"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["fcn", "cnn", "attention"]
WINDOW = 100  # moving average over 100 episodes

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 12, "axes.axisbelow": True,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load_runs(task):
    runs = {}
    prefix = "" if task == "left" else f"{task}_"
    for path in sorted(glob.glob("runs/*/progress.csv")):
        name = os.path.basename(os.path.dirname(path))
        model = name.rsplit("_s", 1)[0]
        if not model.startswith(prefix) or model[len(prefix):] not in MODELS:
            continue
        model = model[len(prefix):]
        df = pd.read_csv(path)
        df["success"] = 100.0 * (df.outcome == "arrived")
        df["crash"] = 100.0 * (df.outcome == "crashed")
        runs.setdefault(model, []).append(df)
    return runs


def band(dfs, column):
    """Moving average per seed, then mean and 95% CI across seeds (episode-aligned)."""
    n = min(len(d) for d in dfs)
    curves = np.stack([d[column].rolling(WINDOW, min_periods=WINDOW // 4).mean().to_numpy()[:n] for d in dfs])
    mean = np.nanmean(curves, 0)
    if len(dfs) > 1:
        ci = 1.96 * np.nanstd(curves, 0, ddof=1) / np.sqrt(len(dfs))
    else:
        ci = np.zeros_like(mean)
    return np.arange(1, n + 1), mean, ci


def training_curves(runs, task):
    panels = [("return", "Total reward per episode"), ("success", "Success rate (%) - " + ("made the left turn" if task == "left" else "reached its exit")),
              ("length", "Episode length (decisions)"), ("speed", "Average speed (m/s)")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    for ax, (col, title) in zip(axes.flat, panels):
        ends = []
        for model in [m for m in ORDER if m in runs]:
            x, mean, ci = band(runs[model], col)
            c = COLORS[model]
            ax.fill_between(x, mean - ci, mean + ci, color=c, alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=c, linewidth=2, label=LABELS[model])
            last = np.where(~np.isnan(mean))[0][-1]
            ends.append([mean[last], x[last], model])
        # direct labels at the line ends, pushed apart so they never overlap
        lo, hi = ax.get_ylim()
        gap = 0.06 * (hi - lo)
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        for y, xe, model in ends:
            ax.annotate(LABELS[model], (xe, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=10, color=INK_2)
        ax.set_title(title)
        ax.set_xlabel("training episode")
        ax.margins(x=0.02)
        ax.set_xlim(right=ax.get_xlim()[1] * 1.12)  # room for the direct labels
    seeds = max(len(v) for v in runs.values())
    axes[0, 0].legend(frameon=False, loc="lower right")
    what = "left turn (paper's task)" if task == "left" else "12 routes, one agent (our extension)"
    fig.suptitle(f"DQN at the intersection, {what} - {WINDOW}-episode moving average, mean of {seeds} seeds "
                 f"(shaded: 95% confidence)", x=0.01, ha="left", fontsize=13, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = f"plots/{'' if task == 'left' else task + '_'}training_curves.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("->", out)


def eval_bars(traffic, task="left"):
    prefix = "" if task == "left" else f"{task}_"
    path = f"results/{prefix}eval_{traffic}.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    order = ["random"] + [m for m in ORDER if m in set(df.model)]
    stats = df.drop(columns=["task"], errors="ignore").groupby("model").agg(["mean", "std"])
    panels = [("success_%", "Success rate (%)"), ("crash_%", "Crash rate (%)"), ("speed", "Average speed (m/s)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, title) in zip(axes, panels):
        means = [stats.loc[m, (col, "mean")] for m in order]
        stds = [0 if np.isnan(stats.loc[m, (col, "std")]) else stats.loc[m, (col, "std")] for m in order]
        bars = ax.bar(range(len(order)), means, width=0.62, color=[COLORS[m] for m in order],
                      yerr=stds, error_kw={"ecolor": INK_2, "elinewidth": 1.2, "capsize": 4})
        for b, m, s in zip(bars, means, stds):
            ax.annotate(f"{m:.0f}" if col != "speed" else f"{m:.1f}", (b.get_x() + b.get_width() / 2, m + s),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=11, color=INK)
        ax.set_xticks(range(len(order)), [LABELS[m] for m in order])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(np.array(means) + np.array(stds)) * 1.18 + 1e-9)
    label = "normal traffic (as in training)" if traffic == "normal" else "DENSE traffic (never seen in training)"
    n = 100 if task == "left" else 150
    fig.suptitle(f"Test on {n} fixed scenarios, {label} - bars: mean over seeds, whiskers: std",
                 x=0.01, ha="left", fontsize=13, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"plots/{prefix}eval_{traffic}.png", dpi=160)
    plt.close(fig)
    print(f"-> plots/{prefix}eval_{traffic}.png")


def turn_bars():
    """Multi-route: success rate per turn type, grouped by turn, one bar per agent (colour = agent)."""
    path = "results/multi_eval_normal.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    turns = ["LEFT", "STRAIGHT", "RIGHT"]
    order = ["random"] + [m for m in ORDER if m in set(df.model)]
    g = df.groupby("model").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(order)
    for i, m in enumerate(order):
        xs = np.arange(len(turns)) + (i - (len(order) - 1) / 2) * width
        vals = [g.loc[m, f"success_{t}_%"] for t in turns]
        bars = ax.bar(xs, vals, width * 0.92, color=COLORS[m], label=LABELS[m])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(turns)), ["Turn LEFT\n(hardest: crosses traffic)", "Go STRAIGHT", "Turn RIGHT\n(easiest)"])
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=len(order), loc="upper left")
    ax.set_title("One agent, all routes: success rate by manoeuvre (normal traffic, mean over seeds)")
    fig.tight_layout()
    fig.savefig("plots/multi_by_turn.png", dpi=160)
    plt.close(fig)
    print("-> plots/multi_by_turn.png")


def main():
    os.makedirs("plots", exist_ok=True)
    for task in ["left", "multi"]:
        runs = load_runs(task)
        if runs:
            training_curves(runs, task)
        eval_bars("normal", task)
        eval_bars("dense", task)
    turn_bars()


if __name__ == "__main__":
    main()
