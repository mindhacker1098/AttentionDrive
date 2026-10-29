"""
Build the presentation: slides/Social_Attention_BanditLAB.pptx (editable in PowerPoint / Google Slides).

    python make_slides.py

Needs plots/ (plot.py), results/ (evaluate.py) and videos/ (demo.py --video ...).
"""
import os

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from common import LABELS

INK, INK_2, ACCENT, SURFACE, RULE = (RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E),
                                     RGBColor(0x2A, 0x78, 0xD6), RGBColor(0xFC, 0xFC, 0xFB),
                                     RGBColor(0xE4, 0xE3, 0xDF))
W, H = Inches(13.333), Inches(7.5)
JOURNEY_NAMES = {"multi_attention": "Ego-Attention, zero-shot", "multi_fcn": "FCN/List, zero-shot",
                 "journey_attention": "Ego-Attention, fine-tuned", "random": "Random"}


def new_slide(prs, title, kicker=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = SURFACE
    if kicker:
        text(s, kicker, Inches(0.6), Inches(0.35), Inches(12), Inches(0.4), 14, ACCENT, bold=True)
    text(s, title, Inches(0.6), Inches(0.7), Inches(12.2), Inches(0.9), 30, INK, bold=True)
    return s


def text(slide, s, x, y, w, h, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, "Segoe UI"
        p.space_after = Pt(6)
    return box


def bullets(slide, items, x, y, w, h, size=20):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        sub = item.startswith("  ")
        r = p.add_run()
        r.text = ("–  " if sub else "•  ") + item.strip()
        r.font.size = Pt(size - 3 if sub else size)
        r.font.color.rgb = INK_2 if sub else INK
        r.font.name = "Segoe UI"
        p.level = 1 if sub else 0
        p.space_after = Pt(8)
    return box


def table(slide, rows, x, y, w, col_widths=None, size=15, row_h=Inches(0.45)):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, row_h * len(rows))
    t = shape.table
    for c, cw in enumerate(col_widths or []):
        t.columns[c].width = cw
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xEE, 0xF4, 0xFC) if r == 0 else SURFACE
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size, run.font.name = Pt(size), "Segoe UI"
                    run.font.bold = r == 0
                    run.font.color.rgb = INK
    return t


def picture(slide, path, x, y, w=None, h=None):
    if os.path.exists(path):
        return slide.shapes.add_picture(path, x, y, w, h)
    text(slide, f"[missing: {path}]", x, y, Inches(5), Inches(0.5), 14, INK_2)


def video(slide, path, poster, x, y, w, h):
    if os.path.exists(path):
        slide.shapes.add_movie(path, x, y, w, h, poster_frame_image=poster if os.path.exists(poster) else None,
                               mime_type="video/mp4")
    else:
        text(slide, f"[missing: {path}]", x, y, Inches(5), Inches(0.5), 14, INK_2)


def eval_rows(path):
    df = pd.read_csv(path)
    order = [m for m in ["random", "fcn", "cnn", "attention"] if m in set(df.model)]
    g = df.groupby("model").mean(numeric_only=True)
    rows = [["Agent", "Success %", "Crash %", "Timeout %", "Avg speed (m/s)"]]
    for m in order:
        rows.append([LABELS[m], f"{g.loc[m, 'success_%']:.0f}", f"{g.loc[m, 'crash_%']:.0f}",
                     f"{g.loc[m, 'timeout_%']:.0f}", f"{g.loc[m, 'speed']:.1f}"])
    return rows


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    L, CW = Inches(0.6), Inches(12.1)

    # 1 - title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = SURFACE
    text(s, "CS6109 Reinforcement Learning  ·  Paper implementation", L, Inches(1.2), CW, Inches(0.5), 16, ACCENT, True)
    text(s, "Teaching a Self-Driving Car to Cross\na Busy Intersection with Attention", L, Inches(1.8), CW,
         Inches(1.9), 40, INK, True)
    text(s, "Leurent & Mercat (INRIA + Renault), “Social Attention for Autonomous Decision-Making in Dense "
            "Traffic”,\nML for Autonomous Driving Workshop, NeurIPS 2019", L, Inches(3.9), Inches(8.4), Inches(1), 18, INK_2)
    text(s, "Bandit LAB\nShakti Prasad Nahak (2611AI46)  ·  Akash Prasad (2611AI08)\n"
            "Gourav Rajak (2611AI07)  ·  Dhruv Kumar Singh (2611AI15)", L, Inches(5.2), Inches(7.6), Inches(1.6), 16, INK)
    picture(s, "slides/frame_title.png", Inches(9.4), Inches(3.9), h=Inches(3.3))

    # 2 - problem
    s = new_slide(prs, "The problem: an unprotected left turn", "1 · Problem")
    bullets(s, ["No traffic light, cars from every direction",
                "Go too early → crash.  Wait too long → never get through",
                "  robots that wait for an empty road are called “freezing robots”",
                "Humans watch the few cars that matter and ignore the rest",
                "Hand-written rules break in new situations → learn by trial and error (RL)"],
            L, Inches(1.8), Inches(6.6), Inches(4.5))
    picture(s, "slides/frame_problem.png", Inches(7.5), Inches(1.7), w=Inches(5.3))

    # 3 - MDP
    s = new_slide(prs, "As a Reinforcement Learning problem", "2 · Formulation")
    table(s, [["RL part", "In this task"],
              ["Agent", "our car"],
              ["Environment", "highway-env intersection simulator (paper's own)"],
              ["State", "1 row per car (us + up to 14): present, x, y, vx, vy, cos θ, sin θ  — order shuffled"],
              ["Actions (every 1 s)", "SLOWER · IDLE · FASTER   (steering is automatic)"],
              ["Reward", "+1 per second at full speed,  −5 crash,  +1 arrival"],
              ["Algorithm", "Double DQN (replay buffer 15k, target net, ε-greedy, γ = 0.95)"]],
          L, Inches(1.8), CW, [Inches(3), Inches(9.1)], size=17, row_h=Inches(0.62))

    # 4 - three networks
    s = new_slide(prs, "Same DQN, three ways to read the scene", "3 · Paper's idea")
    table(s, [["Agent", "How it reads the cars", "Problem"],
              ["FCN/List", "flatten 15×7 table → dense layers [128,128]", "car order is random → can't tell who is who"],
              ["CNN/Grid", "32×32 bird's-eye grid (2 m cells) → small CNN", "coarse picture, mostly empty road"],
              ["Ego-Attention", "each car → encoder; our car asks “who is dangerous for me?”", "—"]],
          L, Inches(1.8), CW, [Inches(2.4), Inches(5.2), Inches(4.5)], size=17, row_h=Inches(0.7))
    text(s, "All three have ~30k weights (paper Table 1) → a fair comparison.  Ours: 30.5k · 31.4k · 34.2k",
         L, Inches(5.0), CW, Inches(0.6), 18, INK_2)

    # 5 - attention
    s = new_slide(prs, "Ego-attention in one example", "3 · Paper's idea")
    table(s, [["Car", "query · key", "softmax weight"],
              ["fast car from the left", "2.0", "0.72"],
              ["far car on the right", "0.5", "0.16"],
              ["car already leaving", "0.0", "0.10"],
              ["ourselves", "−1.5", "0.02"]],
          L, Inches(1.8), Inches(6.2), [Inches(3), Inches(1.5), Inches(1.7)], size=17, row_h=Inches(0.55))
    bullets(s, ["weight = softmax(q·k / √d): sums to 1",
                "Mix the cars' info with these weights → 72% comes from the dangerous car",
                "Same attention as Transformers / ChatGPT",
                "2 heads = two “pairs of eyes”",
                "Order of cars does not matter (permutation invariant)",
                "Any number of cars: empty rows are masked"],
            Inches(7.2), Inches(1.8), Inches(5.6), Inches(4.8), 18)

    # 6 - setup
    s = new_slide(prs, "What we implemented", "4 · Our work")
    bullets(s, ["DQN + all three networks from scratch in PyTorch (~300 lines)",
                "Hyper-parameters copied from the authors' code (rl-agents)",
                "  Double DQN, Adam 5e-4, batch 64, target copy every 512 steps, ε: 1 → 0.05",
                "3 agents × 3 seeds × 4000 episodes = 9 training runs (paper: 120 seeds)",
                "Test: the same 100 unseen traffic scenarios for every agent",
                "Our own experiment: dense traffic never seen in training",
                "  15 starting cars instead of 10, 90% spawn chance instead of 60%",
                "Live demo draws the attention weights as lines (paper Fig. 5)"],
            L, Inches(1.8), CW, Inches(5))

    # 7 - training curves
    s = new_slide(prs, "Training: attention learns the best policy", "5 · Results")
    picture(s, "plots/training_curves.png", Inches(1.0), Inches(1.55), h=Inches(5.8))

    # 8 - test
    s = new_slide(prs, "Test on 100 unseen scenarios", "5 · Results")
    if os.path.exists("results/eval_normal.csv"):
        table(s, eval_rows("results/eval_normal.csv"), L, Inches(1.7), CW, size=16, row_h=Inches(0.45))
    picture(s, "plots/eval_normal.png", Inches(1.4), Inches(4.05), h=Inches(3.3))

    # 9 - dense
    s = new_slide(prs, "Our experiment: heavier traffic than in training", "5 · Results")
    if os.path.exists("results/eval_dense.csv"):
        table(s, eval_rows("results/eval_dense.csv"), L, Inches(1.7), CW, size=16, row_h=Inches(0.45))
    picture(s, "plots/eval_dense.png", Inches(1.4), Inches(4.05), h=Inches(3.3))

    # 10-12 - extension: one agent, 12 routes
    s = new_slide(prs, "Our extension: one agent for all 12 routes", "6 · Beyond the paper")
    bullets(s, ["Paper: one agent, left turn only. Its Section 4 suggests adding the destination as input",
                "We built it: start on any of 4 roads × go left / straight / right = 12 routes",
                "  priority road (west–east) vs. give-way road (north–south): rules of the road differ by route",
                "2 extra numbers per car: direction to its destination (cos d, sin d)",
                "  for our car = the GOAL;  for other cars = their TURN SIGNAL (blinker)",
                "Same DQN, same networks (now 9 features per car); 6000 episodes, 2 seeds",
                "Tested on 150 fixed scenarios, success measured per manoeuvre"],
            L, Inches(1.8), Inches(7.0), Inches(5), 19)
    picture(s, "slides/frame_multi.png", Inches(7.9), Inches(1.7), w=Inches(5.0))

    s = new_slide(prs, "Extension results: learning all routes at once", "6 · Beyond the paper")
    picture(s, "plots/multi_training_curves.png", Inches(1.0), Inches(1.55), h=Inches(5.8))

    s = new_slide(prs, "Extension results by manoeuvre (150 test scenarios)", "6 · Beyond the paper")
    if os.path.exists("results/multi_eval_normal.csv"):
        table(s, eval_rows("results/multi_eval_normal.csv"), L, Inches(1.6), CW, size=15, row_h=Inches(0.42))
    picture(s, "plots/multi_by_turn.png", Inches(2.2), Inches(3.65), h=Inches(3.7))

    # journey
    s = new_slide(prs, "Going further: a long journey through a city", "6 · Beyond the paper")
    bullets(s, ["6 copies of the paper's intersection joined into a 2 × 3 city grid",
                "Random journey to a FINISH line: 3–4 junctions, several turns, ~350 m, up to 26 cars",
                "Route planner (like Google Maps) picks the turns",
                "Our attention DQN decides SLOWER / IDLE / FASTER every second",
                "  at each junction it sees the scene as in training: positions relative to that junction,",
                "  its goal = the exit it must take, other cars' turn signals",
                "Tested zero-shot (never saw the city) and after fine-tuning on it"],
            L, Inches(1.8), Inches(7.0), Inches(5), 19)
    picture(s, "slides/frame_journey.png", Inches(7.9), Inches(1.7), w=Inches(5.0))

    s = new_slide(prs, "Long journey: attention transfers to an unseen city", "6 · Beyond the paper")
    if os.path.exists("results/journey_eval_normal.csv"):
        jd = pd.read_csv("results/journey_eval_normal.csv")
        jd = jd[~jd.agent.str.contains("no blind-spot")]
        jd["kind"] = jd.agent.str.replace(r"_s\d+$", "", regex=True)
        g = jd.groupby("kind").mean(numeric_only=True)
        rows = [["Agent (mean of seeds)", "Finish %", "Crash %", "Too slow %", "Junctions passed"]]
        for kind in ["random", "multi_fcn", "multi_attention", "journey_attention"]:
            if kind in g.index:
                r = g.loc[kind]
                rows.append([JOURNEY_NAMES[kind], f"{r['finish_%']:.0f}", f"{r['crash_%']:.0f}", f"{r['timeout_%']:.0f}",
                             f"{r.junctions_passed:.1f} of {r.junctions_total:.1f}"])
        table(s, rows, L, Inches(1.7), CW, [Inches(4.3), Inches(1.8), Inches(1.8), Inches(1.9), Inches(2.3)],
              size=17, row_h=Inches(0.5))
        bullets(s, ["100 city journeys per agent, identical traffic for every agent",
                    "Attention transfers to the unseen city far better than FCN (FCN crashes in 2 of 3 journeys)",
                    "Small risks add up: ~85% safe per junction → ~57% over 3.4 junctions",
                    "Honest negative: 600 episodes of fine-tuning made it MORE aggressive (speed earns reward every "
                    "second) and it forgot some caution → 44% vs 52.5%"],
                L, Inches(4.55), CW, Inches(2.8), 16)

    s = new_slide(prs, "Demo: the long journey", "7 · Demo")
    video(s, "videos/journey.mp4", "slides/frame_journey.png", L, Inches(1.6), Inches(8.6), Inches(5.5))
    bullets(s, ["Camera follows the car", "Mini-map: whole city + route", "Turn sequence + progress bar",
                "Chequered finish line", "Live: python demo.py --journey"],
            Inches(9.4), Inches(1.8), Inches(3.6), Inches(5), 17)

    # demo
    s = new_slide(prs, "Demo: one agent driving every route", "7 · Demo")
    video(s, "videos/multi_attention.mp4", "slides/frame_multi.png", L, Inches(1.6), Inches(8.6), Inches(5.5))
    bullets(s, ["Green band = planned route", "Lines = attention (thicker = more)", "Amber / blue = the two heads",
                "Tags A–D = most-watched cars", "Panel: Q-value of each action", "Live: python demo.py"],
            Inches(9.4), Inches(1.8), Inches(3.6), Inches(5), 17)

    s = new_slide(prs, "Demo: the paper's left-turn agent", "7 · Demo")
    video(s, "videos/attention.mp4", "slides/frame_attention.png", L, Inches(1.6), Inches(8.6), Inches(5.5))
    bullets(s, ["Lines = attention (thicker = more)", "Amber / blue = the two heads",
                "Ring = attention on itself → “road is clear, go”", "Right panel = Q-value of each action"],
            Inches(9.4), Inches(1.8), Inches(3.6), Inches(5), 17)

    # 11 - baseline video
    s = new_slide(prs, "Compare: the “blind” FCN agent", "7 · Demo")
    video(s, "videos/fcn.mp4", "slides/frame_fcn.png", L, Inches(1.6), Inches(8.6), Inches(5.5))
    bullets(s, ["Same algorithm, same data", "Only the network differs", "Cannot tell which car is which"],
            Inches(9.4), Inches(1.8), Inches(3.6), Inches(5), 17)

    # 12 - conclusion
    s = new_slide(prs, "Conclusion", "8 · Takeaways")
    bullets(s, ["How the network reads the state matters as much as the RL algorithm",
                "Attention: order-free, any number of cars, and interpretable (we can see what it watches)",
                "Our reproduction agrees with the paper's ranking (see results)",
                "We extended it: one agent for 12 routes (goal + turn-signal inputs, the paper's future work)",
                "  and a whole-city journey: attention 52.5% finished vs FCN 34%, without ever training on the city",
                "Limitations: simulator only · rule-based other drivers · speed control only (no lane changes)",
                "  3 seeds vs 120 in the paper (laptop budget) · long journeys: small risks per junction add up",
                "Future: lane changes to pass obstacles, more seeds, real traffic data"],
            L, Inches(1.8), CW, Inches(5))
    text(s, "Code: rl_highway/ (train.py, evaluate.py, demo.py)  ·  Paper: arXiv 1911.12250  ·  Thank you!",
         L, Inches(6.6), CW, Inches(0.5), 16, INK_2)

    os.makedirs("slides", exist_ok=True)
    out = "slides/Social_Attention_BanditLAB.pptx"
    prs.save(out)
    print("->", out)


if __name__ == "__main__":
    main()
