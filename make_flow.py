"""
Flow diagrams of the project (PNG, 16:9, for slides and the report).

    python make_flow.py
-> plots/flow_project.png    whole project: paper -> reproduction -> extensions -> demo
-> plots/flow_training.png   the DQN training loop (what train.py does every step)
-> plots/flow_attention.png  one decision: traffic table -> ego-attention -> Q-values -> action
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
FILL = {"blue": ("#e3eefb", "#2a78d6"), "orange": ("#fde8df", "#eb6834"), "green": ("#dff4ea", "#1baf7a"),
        "grey": ("#efeeeb", "#8a8984"), "violet": ("#ebe8fa", "#6250d6")}


def canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.4, 8.55, title, fontsize=24, fontweight="bold", color=INK, va="center")
    ax.text(0.4, 8.1, subtitle, fontsize=13, color=INK_2, va="center")
    return fig, ax


def box(ax, x, y, w, h, title, body="", color="blue", size=12):
    face, edge = FILL[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=face, edgecolor=edge, linewidth=2))
    if body:
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=size + 1, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.75, body, ha="center", va="top", fontsize=size - 1, color=INK_2, linespacing=1.4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size, fontweight="bold", color=INK,
                linespacing=1.3)
    return (x, y, w, h)


def arrow(ax, a, b, label="", color=INK_2, rad=0.0, lx=0, ly=0.18):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=20, linewidth=2, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((a[0] + b[0]) / 2 + lx, (a[1] + b[1]) / 2 + ly, label, ha="center", va="bottom", fontsize=10.5,
                color=INK_2, style="italic")


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def save(fig, name):
    os.makedirs("plots", exist_ok=True)
    fig.savefig(f"plots/{name}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> plots/{name}.png")


# ----------------------------------------------------------------------------- 1. whole project
def project():
    fig, ax = canvas("Project flow: from the paper to our results",
                     "Bandit LAB · Social Attention for Autonomous Decision-Making in Dense Traffic (NeurIPS 2019 workshop)")
    ax.text(0.4, 7.45, "PART 1 - reproduce the paper", fontsize=13, fontweight="bold", color="#2a78d6")
    w, h, y = 2.35, 2.0, 5.1
    xs = [0.4, 3.0, 5.6, 8.2, 10.8, 13.4]
    row1 = [("Paper", "Leurent & Mercat\nleft turn at a busy\njunction, DQN", "grey"),
            ("Simulator", "highway-env\nintersection-v0\nstate 15 cars × 7", "blue"),
            ("3 networks", "FCN/List\nCNN/Grid\nEgo-Attention (~30k)", "blue"),
            ("DQN training", "Double DQN, replay,\ntarget net, ε-greedy\n3 seeds × 4000 ep.", "blue"),
            ("Testing", "100 unseen scenarios\n+ dense traffic\n(our experiment)", "blue"),
            ("Results", "Attention 67 % / 12 %\nFCN 55 / 41\nCNN 48 / 17", "green")]
    b1 = [box(ax, x, y, w, h, t, body, c) for x, (t, body, c) in zip(xs, row1)]
    for a, b in zip(b1, b1[1:]):
        arrow(ax, right(a), left(b))

    ax.text(0.4, 4.1, "PART 2 - go beyond the paper", fontsize=13, fontweight="bold", color="#eb6834")
    y2 = 1.55
    xs2 = [0.4, 4.35, 8.3, 12.25]
    w2 = 3.6
    row2 = [("Extension 1: 12 routes", "4 roads × left / straight / right\n+ goal & turn-signal inputs\n"
             "70 % / 18 % crash (FCN 65 / 34)", "orange"),
            ("Extension 2: city journey", "6 junctions → finish line\nroute planner picks the turns\n"
             "same agent drives each junction", "orange"),
            ("Journey results", "attention 52.5 % vs FCN 34 %\nfine-tuned: 44 % (too aggressive)\n"
             "risk adds up: 0.85^3.4 ≈ 0.57", "green"),
            ("Demo + presentation", "city graphics, attention lines,\nmini-map, finish line\nvideos + slides + script", "violet")]
    b2 = [box(ax, x, y2, w2, 2.2, t, body, c, size=11.5) for x, (t, body, c) in zip(xs2, row2)]
    for a, b in zip(b2, b2[1:]):
        arrow(ax, right(a), left(b))
    arrow(ax, bottom(b1[2]), top(b2[0]), "same DQN + attention, new tasks", rad=0.0, lx=1.9, ly=-0.05)
    ax.text(0.4, 0.75, "Files: common.py (env + networks) · train.py (DQN) · evaluate.py · journey.py (city) · "
                       "demo.py + city_render.py (demo) · plot.py · make_deck*.py (slides)", fontsize=11, color=INK_2)
    save(fig, "flow_project")


# ----------------------------------------------------------------------------- 2. DQN training loop
def training():
    fig, ax = canvas("Training loop: what happens every step (train.py)",
                     "One decision per second · one gradient update per step · 4000 episodes per run")
    env = box(ax, 0.5, 4.6, 2.9, 1.9, "Environment", "highway-env\nintersection", "grey")
    state = box(ax, 4.3, 4.6, 2.9, 1.9, "State s", "15 cars × 7 numbers\n(positions, speeds,\nheadings)", "blue")
    qnet = box(ax, 8.1, 4.6, 3.0, 1.9, "Q-network θ", "Ego-Attention\n→ Q(s, SLOWER / IDLE /\nFASTER)", "blue")
    act = box(ax, 12.0, 4.6, 3.5, 1.9, "ε-greedy action a", "random with prob. ε\nelse argmax Q\nε: 1 → 0.05", "blue")
    step = box(ax, 12.0, 1.5, 3.5, 1.9, "Step the simulator", "reward r, next state s′\ndone = crash / arrival", "grey")
    buf = box(ax, 8.1, 1.5, 3.0, 1.9, "Replay buffer", "store (s, a, r, s′, done)\nkeep last 15 000\nsample 64 at random",
              "orange")
    tgt = box(ax, 4.3, 1.5, 2.9, 1.9, "Double-DQN target", "y = r + γ · Q_θ̄(s′, a*)\na* = argmax Q_θ(s′)\nγ = 0.95", "orange")
    upd = box(ax, 0.5, 1.5, 2.9, 1.9, "Update θ", "loss = (Q_θ(s,a) − y)²\nAdam, lr 5·10⁻⁴\ngrads clipped ±1", "orange")
    arrow(ax, right(env), left(state), "observe")
    arrow(ax, right(state), left(qnet))
    arrow(ax, right(qnet), left(act))
    arrow(ax, bottom(act), top(step), "act", lx=0.35, ly=-0.1)
    arrow(ax, left(step), right(buf), "store")
    arrow(ax, left(buf), right(tgt), "batch")
    arrow(ax, left(tgt), right(upd), "loss")
    arrow(ax, top(upd), bottom(env), "next step", lx=-0.6, ly=-0.1)
    arrow(ax, (upd[0] + upd[2], upd[1] + upd[3]), (qnet[0], qnet[1]), "improves θ", rad=-0.15, lx=0.4, ly=-0.25,
          color="#eb6834")
    tn = box(ax, 8.4, 6.95, 2.4, 0.75, "Target network θ̄", "", "violet", size=11)
    arrow(ax, top(qnet), bottom(tn), "copy every 512 steps", lx=1.45, ly=-0.18, color="#6250d6")
    ax.text(0.5, 0.65, "Blue = acting (collect experience)   ·   Orange = learning (update the network)   ·   "
                       "Violet = slowly updated copy that keeps the target steady", fontsize=11.5, color=INK_2)
    save(fig, "flow_training")


# ----------------------------------------------------------------------------- 3. one decision with attention
def attention():
    fig, ax = canvas("One decision: how Ego-Attention turns the traffic into an action",
                     "common.py · EgoAttentionNet.attend() + decoder · d = 64, 2 heads of 32")
    w, h, y = 2.25, 2.3, 4.6
    steps = [("Traffic table", "row 0 = our car\nrows 1-14 = other cars\n(shuffled order)", "grey"),
             ("Encoders", "small shared network\nper car → e_i\n(64 numbers)", "blue"),
             ("Q, K, V", "query q from our car\nkey k_i, value v_i\nfrom every car", "blue"),
             ("Scores", "q · k_i / √32\nempty rows → −∞", "blue"),
             ("Softmax α", "weights that\nadd up to 1\n(who matters?)", "orange"),
             ("Mix + decode", "z = Σ α_i v_i\n+ our own e_0\n→ decoder", "blue")]
    xs = [0.3 + i * 2.6 for i in range(6)]
    bs = []
    for x, (t, body, c) in zip(xs, steps):
        b = box(ax, x, y, w, h, t, body, c, size=11.5)
        bs.append(b)
    for a, b in zip(bs, bs[1:]):
        arrow(ax, right(a), left(b))
    q = box(ax, 8.1, 1.3, 3.2, 2.1, "Q-values", "SLOWER  1.33\nIDLE        1.43  ← max\nFASTER    1.09", "green")
    a = box(ax, 12.4, 1.3, 3.1, 2.1, "Action", "argmax Q → IDLE\n(sent to the car,\nonce per second)", "green")
    arrow(ax, bottom(bs[5]), top(q), rad=-0.2)
    arrow(ax, right(q), left(a))
    ex = box(ax, 0.3, 1.3, 7.0, 2.1, "Example: attention weights",
             "fast car from the left 0.72  ·  far car 0.16\ncar leaving 0.10  ·  ourselves 0.02\n"
             "→ the dangerous car drives the decision\n→ drawn as lines in the demo",
             "orange", size=11.5)
    arrow(ax, bottom(bs[4]), (ex[0] + ex[2] - 0.6, ex[1] + ex[3]), rad=0.0, color="#eb6834")
    ax.text(0.3, 0.6, "Order-free: re-ordering rows 1-14 re-orders the α_i and v_i together, so the sum z (and the action) "
                      "stays the same.", fontsize=11.5, color=INK_2)
    save(fig, "flow_attention")


if __name__ == "__main__":
    project()
    training()
    attention()
