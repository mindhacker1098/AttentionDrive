"""
Final presentation for the TA: full mechanism (RL maths, the paper's attention model, our code), results,
extensions and demo - with a word-for-word script for 3 speakers.

    python make_deck.py

-> slides/BanditLAB_Final_Presentation.pptx   (the script is also in each slide's speaker notes:
                                                PowerPoint > Slide Show > Use Presenter View shows it to the speaker)
-> slides/BanditLAB_Presentation_Script.docx  (printable script, one section per speaker)
Main part: slides 1-18, about 10 minutes. Appendix A1-A5: backup slides for questions (no script).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt as DPt, RGBColor as DRGB
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

INK, INK_2, ACCENT, SURFACE = RGBColor(11, 11, 11), RGBColor(82, 81, 78), RGBColor(42, 120, 214), RGBColor(252, 252, 251)
SPEAKER_COLORS = {1: RGBColor(42, 120, 214), 2: RGBColor(0, 131, 0), 3: RGBColor(201, 90, 30)}
L, CW = Inches(0.6), Inches(12.1)
EQ_DIR = "slides/eq"

# ----------------------------------------------------------------------------- the script (single source of truth)
# slide number -> (speaker, seconds, words to say)
SCRIPT = {
    1: (1, 25, "Good afternoon. We are Bandit LAB. Our project implements the paper “Social Attention for Autonomous "
               "Decision-Making in Dense Traffic” by Edouard Leurent and Jean Mercat, from INRIA and Renault, published at "
               "the NeurIPS 2019 workshop on machine learning for autonomous driving. We reproduced the paper, and then we "
               "extended it in two ways."),
    2: (1, 30, "The problem is an unprotected left turn. There is no traffic light and cars come from every direction. "
               "If our car goes too early, it crashes. If it waits too long, it never gets through. This is called the "
               "freezing robot problem. Hand-written rules do not cover every situation, so we let the car learn by trial "
               "and error. That is reinforcement learning."),
    3: (1, 40, "We write the task as a Markov Decision Process. The state is a table with one row per car: our car plus "
               "up to fourteen others. Each row has seven numbers: present or not, position x and y, velocity x and y, "
               "and the cosine and sine of the heading. Positions are divided by one hundred and velocities by twenty, so "
               "every number is between minus one and one. Once per second the agent chooses slower, idle or faster; "
               "steering is automatic. The reward is between zero and one per second depending on speed, minus five for a "
               "crash, and plus one for reaching the exit."),
    4: (1, 35, "The goal is to maximise the return: the sum of future rewards, where a reward k steps ahead is "
               "multiplied by gamma to the power k. We use gamma zero point nine five, so the agent looks about twenty "
               "seconds ahead. Q of s and a is the expected return after taking action a in state s. The best Q satisfies "
               "the Bellman optimality equation: value now equals reward plus gamma times the best value in the next state. "
               "Once we know Q, the policy is simply: pick the action with the highest Q."),
    5: (1, 50, "The state is continuous, so we cannot store Q in a table. DQN approximates Q with a neural network with "
               "weights theta. We train it towards the target y: reward plus gamma times the value of the next state, using "
               "the squared error as loss. Three tricks make this stable. One: a replay buffer of fifteen thousand "
               "transitions; we learn from random batches of sixty-four. Two: a target network, a frozen copy of the "
               "weights, updated every 512 steps. Three: Double DQN: the online network picks the next action and the "
               "target network scores it, which reduces over-estimation. Epsilon-greedy exploration starts at one and "
               "decays to zero point zero five. The code on the right is exactly these lines from our train dot py. "
               "Now my teammate will explain the paper's main idea."),
    6: (2, 30, "The paper keeps DQN fixed and asks one question: how should the network read the other cars? A plain list "
               "has two problems: its size changes with the number of cars, and it depends on the order of the cars. But "
               "a good driving policy must be permutation invariant: if we only re-order the other cars, the decision must "
               "not change. That is the equation on this slide. Like the paper, we shuffle the order of the cars at every "
               "step."),
    7: (2, 30, "We compare three networks with about thirty thousand weights each, so the comparison is fair. FCN List "
               "flattens the table into one hundred and five numbers and uses two dense layers; it is not order invariant. "
               "CNN Grid draws a thirty-two by thirty-two bird's-eye grid with two-metre cells; it is order invariant but "
               "coarse. Ego-Attention is the paper's model. Our parameter counts match the paper's Table 1."),
    8: (2, 45, "This is the attention mechanism step by step. Each car's row goes through a small shared encoder and "
               "becomes a sixty-four number vector e. Our car, the ego car, makes a query q. Every car makes a key k and a "
               "value v. The score of car i is q dot k i, divided by the square root of the head size, thirty-two. A "
               "softmax turns the scores into weights alpha that add up to one; empty rows are masked out. The output is "
               "the weighted sum of the values. We use two heads, so this happens twice in parallel. The result is added "
               "to the ego encoding, and a decoder outputs the three Q-values."),
    9: (2, 30, "A small example. The scores are two for a fast car from the left, zero point five for a far car, zero for "
               "a car leaving, and minus one point five for ourselves. Softmax gives zero point seven two, zero point one "
               "six, zero point one and zero point zero two. So seventy-two percent of what the network uses comes from "
               "the dangerous car. Because the output is a weighted sum, re-ordering the cars does not change it, and it "
               "works for any number of cars."),
    10: (2, 30, "This is our attend function in common dot py. Each line matches an equation: encoders, the query from "
                "the ego car, keys and values from everyone, scaled dot product, the mask, softmax, weighted sum and the "
                "residual. A bonus is that we can see the attention: we draw the weights as lines from our car, as in the "
                "picture. Now our third member will present the experiments and results."),
    11: (3, 25, "We implemented everything from scratch in PyTorch and copied the hyper-parameters from the authors' own "
                "code. Three agents, three seeds each, four thousand episodes per run: nine runs. The paper used one "
                "hundred and twenty seeds, so we show ninety-five percent confidence bands. For testing, every agent "
                "drives the same one hundred unseen scenarios with no exploration."),
    12: (3, 25, "Our training curves show the same three behaviours as the paper's Figure 4. FCN, in orange, is the "
                "fastest but crashes and has the shortest episodes: risky and blind. CNN, in green, is the slowest: too "
                "cautious. Ego-Attention, in blue, gets the highest reward: it waits when needed and uses gaps."),
    13: (3, 25, "On the test, attention succeeds in sixty-seven percent of scenarios with only twelve percent crashes. "
                "FCN crashes forty-one percent of the time, and CNN times out thirty-four percent of the time. With denser "
                "traffic than in training, the ranking stays the same: FCN then crashes sixty-one percent of the time."),
    14: (3, 35, "Then we went beyond the paper. Its Section 4 suggests giving the agent its destination. We did that: our "
                "car starts on any of four roads and goes left, straight or right: twelve routes. Each car gets the "
                "direction to its destination; for our car it is the goal, for other cars it acts like a turn signal. One "
                "attention agent learned all twelve routes and halved the crash rate compared with FCN, eighteen versus "
                "thirty-four percent, with the biggest gain on the left turn."),
    15: (3, 40, "Our second extension is a long journey through a city of six intersections to a finish line. A route "
                "planner picks the turns with a shortest-path search, and our attention agent decides when to go. At each "
                "junction we give it positions relative to that junction, so its skill transfers. Without any training in "
                "the city, attention finished fifty-two and a half percent of journeys, FCN only thirty-four. Risks add "
                "up: eighty-five percent per junction becomes about fifty-seven percent per journey. Fine-tuning on the "
                "city made it worse, because it became more aggressive. We report that honestly."),
    16: (3, 30, "Here is our demo. The green band is the planned route. The lines show attention: thicker means more "
                "attention, and the two colours are the two heads. On the right are the Q-values and the chosen action. "
                "The camera follows the car to the finish line."),
    17: (3, 25, "To conclude: how the network reads the state matters as much as the learning algorithm. Attention is "
                "order-free, works for any number of cars, and shows what the agent is looking at. We reproduced the "
                "paper and extended it to twelve routes and a whole city. Limitations: it is a simulator, other drivers "
                "follow simple rules, and our car only controls its speed."),
    18: (3, 5, "Thank you. We are happy to take your questions."),
}


# ----------------------------------------------------------------------------- equations as images
def eq(name, latex, size=26):
    os.makedirs(EQ_DIR, exist_ok=True)
    path = os.path.join(EQ_DIR, name + ".png")
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=size, color="#0b0b0b")
    fig.savefig(path, dpi=250, transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return path


EQS = {
    "return": r"G_t=\sum_{k=0}^{\infty}\gamma^{k}\,r_{t+k},\qquad \gamma=0.95",
    "qdef": r"Q^{\pi}(s,a)=\mathbb{E}\left[\,G_t \mid s_t=s,\ a_t=a\,\right]",
    "bellman": r"Q^{*}(s,a)=\mathbb{E}_{s'}\left[\,r(s,a)+\gamma\,\max_{a'}Q^{*}(s',a')\,\right]",
    "policy": r"\pi(s)=\arg\max_{a}\ Q(s,a)",
    "reward": r"r_t=\mathrm{clip}\left(\frac{v_t-7}{2},\,0,\,1\right)-5\cdot\mathbf{1}[\mathrm{crash}]\qquad (r_t=+1\ \mathrm{on\ arrival})",
    "target": r"y=r+\gamma\,(1-d)\;Q_{\bar{\theta}}\left(s',\ \arg\max_{a'}Q_{\theta}(s',a')\right)",
    "loss": r"\mathcal{L}(\theta)=\frac{1}{B}\sum_{j=1}^{B}\left(Q_{\theta}(s_j,a_j)-y_j\right)^{2}",
    "eps": r"\epsilon_t=0.05+0.95\,e^{-t/15000}",
    "perm": r"\pi(\cdot\mid s_0,s_1,\ldots,s_N)=\pi(\cdot\mid s_0,s_{\tau(1)},\ldots,s_{\tau(N)})\quad\forall\,\tau",
    "enc": r"e_i=f(s_i),\qquad q=W_Q\,e_0,\qquad k_i=W_K\,e_i,\qquad v_i=W_V\,e_i",
    "alpha": r"\alpha_i=\frac{\exp(q\cdot k_i/\sqrt{d_h})}{\sum_{j}\exp(q\cdot k_j/\sqrt{d_h})},\qquad d_h=32",
    "out": r"z=\sum_{i}\alpha_i\,v_i,\qquad o=\frac{1}{2}\left(W_O\,z+e_0\right),\qquad Q(s,\cdot)=g(o)\in\mathbb{R}^{3}",
    "goal": r"d_i=\frac{g_i-p_i}{\|g_i-p_i\|}\quad(\cos d,\ \sin d)",
    "compound": r"P(\mathrm{journey})=\prod_{j=1}^{n}p_j\ \approx\ 0.85^{3.4}\ \approx\ 0.57",
    "local": r"\tilde{x}_i=\frac{p_i-c_k}{100}",
}

# ----------------------------------------------------------------------------- code excerpts (copied from our files)
CODE_DQN = """# train.py - one DQN update per environment step
s, a, r, s2, term = buffer.sample(64, device)
q = q_net(s).gather(1, a[:, None]).squeeze(1)
with torch.no_grad():   # Double DQN
    best = q_net(s2).argmax(1, keepdim=True)
    target = r + gamma * (1 - term) * \\
             target_net(s2).gather(1, best).squeeze(1)
loss = F.mse_loss(q, target)
optimizer.zero_grad(); loss.backward()
for prm in q_net.parameters():
    prm.grad.data.clamp_(-1, 1)      # gradient clipping
optimizer.step()
if step % 512 == 0:                  # target network
    target_net.load_state_dict(q_net.state_dict())"""

CODE_ATT = """# common.py - EgoAttentionNet.attend()
ego = self.ego_encoder(x[:, :1])                       # e_0
everyone = torch.cat([ego, self.others_encoder(x[:, 1:])], 1)
q = self.query(ego).view(B, 1, h, dh).transpose(1, 2)  # q
k = self.key(everyone).view(B, N, h, dh).transpose(1, 2)
v = self.value(everyone).view(B, N, h, dh).transpose(1, 2)
scores = q @ k.transpose(-2, -1) / dh ** 0.5           # q.k/sqrt(dh)
absent = (x[:, :, 0] < 0.5).view(B, 1, 1, N)           # empty rows
attention = scores.masked_fill(absent, -1e9).softmax(-1)
mixed = (attention @ v).reshape(B, self.d)             # sum a_i v_i
out = (self.combine(mixed) + ego.squeeze(1)) / 2       # residual
return out, attention.squeeze(2)"""


# ----------------------------------------------------------------------------- pptx helpers
def text(slide, s, x, y, w, h, size=18, color=INK, bold=False, font="Segoe UI", align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, font
        p.space_after = Pt(4)
    return box


def bullets(slide, items, x, y, w, h, size=19):
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
        p.space_after = Pt(7)
    return box


def code(slide, src, x, y, w, h, size=12):
    box = slide.shapes.add_textbox(x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(28, 30, 36)
    tf = box.text_frame
    tf.word_wrap = False
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(0.15))
    for i, line in enumerate(src.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = line
        comment = line.strip().startswith("#")
        r.font.size, r.font.name = Pt(size), "Consolas"
        r.font.color.rgb = RGBColor(130, 200, 140) if comment else RGBColor(232, 232, 228)
    return box


def table(slide, rows, x, y, w, col_widths=None, size=15, row_h=Inches(0.42)):
    t = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, row_h * len(rows)).table
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
                    run.font.size, run.font.name, run.font.bold = Pt(size), "Segoe UI", r == 0
                    run.font.color.rgb = INK
    return t


def picture(slide, path, x, y, w=None, h=None):
    if os.path.exists(path):
        return slide.shapes.add_picture(path, x, y, w, h)
    text(slide, f"[missing: {path}]", x, y, Inches(5), Inches(0.5), 14, INK_2)


def equation(slide, name, x, y, h=Inches(0.6)):
    return slide.shapes.add_picture(eq(name, EQS[name]), x, y, height=h)


def new_slide(prs, no, title, kicker):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = SURFACE
    text(s, kicker, L, Inches(0.32), Inches(9), Inches(0.4), 14, ACCENT, bold=True)
    text(s, title, L, Inches(0.68), Inches(12.2), Inches(0.9), 28, INK, bold=True)
    if no in SCRIPT:
        spk, secs, words = SCRIPT[no]
        text(s, f"Speaker {spk}", Inches(11.3), Inches(0.3), Inches(1.6), Inches(0.4), 13,
             SPEAKER_COLORS[spk], bold=True, align=PP_ALIGN.RIGHT)
        s.notes_slide.notes_text_frame.text = f"SPEAKER {spk}  (~{secs} s)\n\n{words}"
    return s


# ----------------------------------------------------------------------------- the deck
def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # 1 title
    s = new_slide(prs, 1, "", "CS6109 Reinforcement Learning  ·  Paper implementation")
    text(s, "Teaching a Self-Driving Car\nto Cross a Busy Intersection\nwith Attention", L, Inches(1.0), Inches(8.4),
         Inches(2.2), 34, INK, True)
    text(s, "Paper: Leurent & Mercat (INRIA + Renault), “Social Attention for Autonomous Decision-Making in Dense "
            "Traffic”, ML for Autonomous Driving Workshop, NeurIPS 2019", L, Inches(3.45), Inches(8.4), Inches(1), 17, INK_2)
    text(s, "Reproduction  +  12-route extension  +  long journey through a city", L, Inches(4.4), Inches(8.4),
         Inches(0.5), 18, ACCENT, True)
    text(s, "Bandit LAB\nShakti Prasad Nahak (2611AI46)  ·  Akash Prasad (2611AI08)\n"
            "Gourav Rajak (2611AI07)  ·  Dhruv Kumar Singh (2611AI15)", L, Inches(5.2), Inches(8), Inches(1.6), 16, INK)
    picture(s, "slides/frame_title.png", Inches(9.2), Inches(1.5), h=Inches(3.9))

    # 2 problem
    s = new_slide(prs, 2, "The problem: an unprotected left turn", "1 · Problem")
    bullets(s, ["No traffic light, cars from every direction",
                "Go too early → crash",
                "Wait too long → never get through (“freezing robot”)",
                "Humans watch the few cars that matter and ignore the rest",
                "Hand-written rules do not cover every case → learn by trial and error = Reinforcement Learning"],
            L, Inches(1.8), Inches(6.5), Inches(4.8), 20)
    picture(s, "slides/frame_problem.png", Inches(7.4), Inches(1.6), h=Inches(5.5))

    # 3 MDP
    s = new_slide(prs, 3, "The task as a Markov Decision Process (S, A, P, R, γ)", "2 · Reinforcement learning")
    table(s, [["Part", "In our task"],
              ["State s", "table: 15 cars × 7 numbers (present, x, y, vx, vy, cos θ, sin θ), row 0 = our car; "
                          "x,y ÷ 100, v ÷ 20 → all in [−1, 1]; other cars' order shuffled every step"],
              ["Action a", "every 1 s: SLOWER · IDLE · FASTER (target speed 0 / 4.5 / 9 m/s); steering is automatic"],
              ["Transition P", "highway-env simulator (15 physics steps per decision), other drivers = IDM model"],
              ["Episode", "ends on crash, on arrival, or after 13 s"],
              ["Discount γ", "0.95 → looks ≈ 1/(1−γ) = 20 decisions ahead"]],
          L, Inches(1.7), CW, [Inches(2.2), Inches(9.9)], size=15, row_h=Inches(0.55))
    text(s, "Reward (highway-env intersection env):", L, Inches(5.35), Inches(6), Inches(0.4), 16, INK_2, True)
    equation(s, "reward", L, Inches(5.85), h=Inches(0.62))

    # 4 return / Q / Bellman
    s = new_slide(prs, 4, "What the agent maximises: return, Q-value, Bellman equation", "2 · Reinforcement learning")
    rows = [("Return (sum of discounted future rewards):", "return"),
            ("Action value = expected return after taking a in s:", "qdef"),
            ("Bellman optimality equation (paper, Section 2):", "bellman"),
            ("Policy = act greedily with respect to Q:", "policy")]
    for i, (label, name) in enumerate(rows):
        y = Inches(1.65 + 1.35 * i)
        text(s, label, L, y, Inches(11), Inches(0.4), 16, INK_2, True)
        equation(s, name, Inches(1.0), y + Inches(0.42), h=Inches(0.85 if name == "return" else 0.7))

    # 5 DQN
    s = new_slide(prs, 5, "DQN: approximate Q with a neural network", "2 · Reinforcement learning")
    text(s, "Double-DQN target and loss:", L, Inches(1.55), Inches(6.5), Inches(0.4), 16, INK_2, True)
    equation(s, "target", L, Inches(1.95), h=Inches(0.62))
    equation(s, "loss", L, Inches(2.7), h=Inches(0.75))
    bullets(s, ["Replay buffer (15 000): learn from random batches of 64 → breaks step-to-step correlation",
                "Target network θ̄: frozen copy, refreshed every 512 steps → a steady target",
                "Double DQN: online net picks a′, target net scores it → less over-estimation",
                "d = 1 only for a real end (crash / arrival) · Adam, lr 5·10⁻⁴ · gradients clipped to ±1"],
            L, Inches(3.6), Inches(6.6), Inches(2.9), 15)
    text(s, "ε-greedy exploration:", L, Inches(6.3), Inches(2.6), Inches(0.5), 15, INK_2, True)
    equation(s, "eps", Inches(3.2), Inches(6.22), h=Inches(0.45))
    code(s, CODE_DQN, Inches(7.35), Inches(1.55), Inches(5.6), Inches(5.3), 11)

    # 6 permutation invariance
    s = new_slide(prs, 6, "How should the network read the other cars?", "3 · The paper's idea")
    bullets(s, ["Same DQN for every agent - only the network that reads the state changes",
                "List of features: size changes with the number of cars, and it depends on their order",
                "Car #3 is a different car every step (we shuffle the rows, like the paper)",
                "A good driving policy must give the same decision for any order of the other cars:"],
            L, Inches(1.8), CW, Inches(3), 20)
    equation(s, "perm", Inches(1.2), Inches(4.3), h=Inches(0.65))
    text(s, "τ = any re-ordering of cars 1…N  ·  “permutation invariant” (paper eq. 2)", Inches(1.2), Inches(5.15),
         Inches(10), Inches(0.5), 16, INK_2)

    # 7 three networks
    s = new_slide(prs, 7, "Same DQN, three networks (≈ 30 k weights each)", "3 · The paper's idea")
    table(s, [["Agent", "Input", "Layers", "Weights (ours)", "Order-free?"],
              ["FCN/List", "15×7 table, flattened (105)", "105 → 128 → 128 → 3", "30 467", "no"],
              ["CNN/Grid", "32×32 grid, 2 m cells, 7 channels", "3 conv (k2, s2): 16-32-64 → 20 → 3", "31 363", "yes (coarse)"],
              ["Ego-Attention", "15×7 table, any order", "enc 7→64→64 · 2 heads · dec 64→64→64→3", "34 243", "yes"]],
          L, Inches(1.8), CW, [Inches(2.2), Inches(3.2), Inches(3.9), Inches(1.6), Inches(1.2)], size=16,
          row_h=Inches(0.62))
    text(s, "Paper Table 1: 3.0e4 · 3.2e4 · 3.4e4  →  our networks match.", L, Inches(4.6), CW, Inches(0.5), 17, INK_2)

    # 8 attention maths
    s = new_slide(prs, 8, "Ego-attention, step by step", "3 · The paper's idea")
    steps = [("1. Encode every car with a shared small network; our car asks, every car answers:", "enc"),
             ("2. Scaled dot-product scores → softmax (empty rows masked) → weights that sum to 1:", "alpha"),
             ("3. Weighted sum of values, residual with our own encoding, decoder → 3 Q-values:", "out")]
    for i, (label, name) in enumerate(steps):
        y = Inches(1.65 + 1.55 * i)
        text(s, label, L, y, Inches(12), Inches(0.4), 16, INK_2, True)
        equation(s, name, Inches(1.0), y + Inches(0.45), h=Inches({"enc": 0.5, "alpha": 0.95, "out": 0.8}[name]))
    text(s, "2 heads (d = 64, d_h = 32) run in parallel and are concatenated before W_O.  "
            "“Ego”: only our car asks the query.", L, Inches(6.45), CW, Inches(0.5), 15, INK_2)

    # 9 worked example
    s = new_slide(prs, 9, "Attention in one example", "3 · The paper's idea")
    table(s, [["Car", "q·k / √d_h", "softmax weight α"],
              ["fast car from the left", "2.0", "0.72"],
              ["far car on the right", "0.5", "0.16"],
              ["car already leaving", "0.0", "0.10"],
              ["ourselves", "−1.5", "0.02"]],
          L, Inches(1.8), Inches(6.4), [Inches(3.2), Inches(1.5), Inches(1.7)], size=17, row_h=Inches(0.55))
    text(s, "e² = 7.39,  e^0.5 = 1.65,  e⁰ = 1,  e^−1.5 = 0.22  →  sum 10.26\n7.39 / 10.26 = 0.72", L,
         Inches(4.8), Inches(6.4), Inches(1), 15, INK_2)
    bullets(s, ["72 % of what the network uses comes from the dangerous car",
                "Weighted SUM → re-ordering the cars cannot change it (order-free)",
                "Works for any number of cars (empty rows get weight 0)",
                "Same attention as Transformers / ChatGPT, applied to cars"],
            Inches(7.3), Inches(1.8), Inches(5.6), Inches(4.5), 18)

    # 10 attention code
    s = new_slide(prs, 10, "The same thing in our code", "3 · The paper's idea")
    code(s, CODE_ATT, L, Inches(1.6), Inches(7.6), Inches(3.6), 12)
    picture(s, "slides/frame_multi.png", Inches(8.45), Inches(1.6), w=Inches(4.5))
    text(s, "Attention weights drawn as lines from our car (thicker = more): amber = head 1, blue = head 2",
         Inches(8.45), Inches(4.6), Inches(4.5), Inches(1), 14, INK_2)

    # 11 setup
    s = new_slide(prs, 11, "What we implemented and how we tested", "4 · Experiments")
    table(s, [["Setting", "Value (from the authors' rl-agents configs)"],
              ["Algorithm", "Double DQN, MSE loss, Adam lr 5·10⁻⁴, batch 64, replay 15 000, target copy every 512 steps"],
              ["Exploration", "ε = 0.05 + 0.95·e^(−t/15000)"],
              ["Training", "3 agents × 3 seeds × 4000 episodes = 9 runs (paper: 120 seeds)"],
              ["Test", "100 unseen scenarios, the same for every agent, no exploration (greedy)"],
              ["Our experiment", "dense traffic never seen in training: 15 starting cars (not 10), 90 % spawn (not 60 %)"],
              ["Code", "PyTorch from scratch: common.py · train.py · evaluate.py · plot.py · demo.py"]],
          L, Inches(1.7), CW, [Inches(2.4), Inches(9.7)], size=15, row_h=Inches(0.58))

    # 12 training curves
    s = new_slide(prs, 12, "Training: the same three behaviours as the paper", "5 · Results")
    picture(s, "plots/training_curves.png", L, Inches(1.5), h=Inches(5.85))
    bullets(s, ["FCN: fastest, shortest episodes → crashes (“risky and blind”)", "CNN: slowest (“too cautious”)",
                "Attention: highest reward 5.7 vs 4.3 vs 2.8"], Inches(10.3), Inches(1.9), Inches(2.8), Inches(5), 15)

    # 13 test
    s = new_slide(prs, 13, "Test on 100 unseen scenarios", "5 · Results")
    table(s, [["Agent", "Success %", "Crash %", "Too slow %", "Dense traffic: success / crash %"],
              ["Random", "20", "40", "40", "15 / 54"],
              ["FCN/List", "55", "41", "4", "32 / 61"],
              ["CNN/Grid", "48", "17", "34", "27 / 25"],
              ["Ego-Attention", "67", "12", "21", "42 / 17"]],
          L, Inches(1.6), CW, [Inches(2.6), Inches(1.9), Inches(1.9), Inches(1.9), Inches(3.8)], size=16,
          row_h=Inches(0.45))
    picture(s, "plots/eval_normal.png", Inches(1.3), Inches(4.0), h=Inches(3.35))

    # 14 extension 1
    s = new_slide(prs, 14, "Extension 1: one agent for all 12 routes", "6 · Beyond the paper")
    bullets(s, ["Paper: left turn only; its Section 4 suggests adding the destination",
                "Start on any of 4 roads × go left / straight / right = 12 routes",
                "2 extra numbers per car = direction to its destination:"],
            L, Inches(1.7), Inches(6.4), Inches(2.4), 17)
    equation(s, "goal", Inches(1.0), Inches(3.35), h=Inches(0.75))
    bullets(s, ["our car → its GOAL;  other cars → their TURN SIGNAL",
                "Result: attention 70 % success / 18 % crash vs FCN 65 % / 34 %",
                "Biggest gain on the left turn: 52 % vs 41 %"],
            L, Inches(4.35), Inches(6.4), Inches(2.6), 17)
    picture(s, "plots/multi_by_turn.png", Inches(7.0), Inches(1.9), w=Inches(6.0))

    # 15 extension 2
    s = new_slide(prs, 15, "Extension 2: a long journey through a city", "6 · Beyond the paper")
    bullets(s, ["6 intersections in a 2 × 3 grid, finish line 3–4 junctions away",
                "Route planner (breadth-first shortest path) chooses the turns",
                "Attention DQN chooses SLOWER / IDLE / FASTER, seeing each junction as in training:"],
            L, Inches(1.65), Inches(7.2), Inches(2), 16)
    equation(s, "local", Inches(1.0), Inches(3.25), h=Inches(0.7))
    text(s, "c_k = centre of the junction ahead\ngoal = the exit to take there", Inches(3.3), Inches(3.3),
         Inches(4.2), Inches(0.7), 14, INK_2)
    table(s, [["100 journeys, same traffic", "Finish %", "Crash %"],
              ["Random", "37", "43"], ["FCN, zero-shot", "34", "66"],
              ["Attention, zero-shot", "52.5", "45.5"], ["Attention, fine-tuned", "44", "55.5"]],
          L, Inches(4.15), Inches(7.0), [Inches(4.0), Inches(1.5), Inches(1.5)], size=15, row_h=Inches(0.4))
    equation(s, "compound", L, Inches(6.35), h=Inches(0.62))
    picture(s, "slides/frame_journey.png", Inches(7.9), Inches(1.65), w=Inches(5.1))

    # 16 demo
    s = new_slide(prs, 16, "Demo: the agent drives to the finish line", "7 · Demo")
    if os.path.exists("videos/journey.mp4"):
        s.shapes.add_movie("videos/journey.mp4", L, Inches(1.5), Inches(9.1), Inches(5.83),
                           poster_frame_image="slides/frame_journey.png", mime_type="video/mp4")
    bullets(s, ["Green band = planned route", "Lines = attention (thicker = more)", "Amber / blue = 2 heads",
                "Panel: Q-values + chosen action", "Mini-map + finish line",
                "Live: python demo.py --run runs/multi_attention_s2 --journey"],
            Inches(9.9), Inches(1.7), Inches(3.2), Inches(5.5), 15)

    # 17 conclusion
    s = new_slide(prs, 17, "Conclusion", "8 · Takeaways")
    bullets(s, ["How the network reads the state matters as much as the RL algorithm",
                "Attention: order-free, any number of cars, and we can see what it watches",
                "Reproduced the paper: attention 67 % success / 12 % crash vs FCN 55 / 41, CNN 48 / 17",
                "Extended it: one agent for 12 routes (crashes halved) and a whole-city journey (52.5 % vs 34 %)",
                "Limitations: simulator · rule-based other drivers · speed control only (no lane changes) · 3 seeds",
                "Future: lane changes to pass obstacles, more seeds, longer fine-tuning, real traffic data"],
            L, Inches(1.8), CW, Inches(5), 19)

    # 18 thanks
    s = new_slide(prs, 18, "Thank you - questions?", "")
    text(s, "Code: rl_highway/  (common.py · train.py · evaluate.py · journey.py · demo.py)\n"
            "Paper: arXiv 1911.12250  ·  Authors' configs: github.com/eleurent/rl-agents", L, Inches(2.0), CW,
         Inches(1.2), 18, INK_2)
    picture(s, "slides/frame_attention.png", Inches(3.4), Inches(3.3), h=Inches(3.9))

    # ---------------- appendix (backup for questions)
    s = new_slide(prs, 101, "A1 · Code map: where each part lives", "Appendix")
    table(s, [["File", "Key parts", "What it does"],
              ["common.py", "make_env, LIST_OBS, GRID_OBS", "environments (paper task, 12 routes), observation settings"],
              ["common.py", "FCNList, GridCNN, EgoAttentionNet", "the three networks (attend() = attention maths)"],
              ["common.py", "ReplayBuffer, epsilon_at", "experience memory and exploration schedule"],
              ["train.py", "main loop", "DQN training: act ε-greedy, store, sample, Double-DQN update, target copy"],
              ["evaluate.py", "evaluate()", "100 fixed test scenarios, greedy policy, success / crash / timeout"],
              ["journey.py", "JourneyEnv, local_obs, current_junction", "6-junction city, route planner, junction-relative view"],
              ["demo.py / city_render.py", "Demo, CityRenderer", "live demo, attention lines, city graphics"],
              ["plot.py / make_deck.py", "-", "figures and this presentation"]],
          L, Inches(1.6), CW, [Inches(2.6), Inches(3.8), Inches(5.7)], size=14, row_h=Inches(0.5))

    s = new_slide(prs, 102, "A2 · Why the target network and Double DQN?", "Appendix")
    bullets(s, ["Without a target network the target y moves every time θ changes → the network chases itself",
                "With θ̄ fixed for 512 steps, each update is plain supervised regression towards a steady y",
                "max over noisy Q estimates is biased upwards (max of noise > 0) → plain DQN over-estimates",
                "Double DQN: choose a′ with θ, evaluate it with θ̄ → the two errors are independent, bias shrinks",
                "Truncation (13 s limit) is not a real end: we still bootstrap there (d = 0); crash / arrival → d = 1"],
            L, Inches(1.8), CW, Inches(4.5), 18)

    s = new_slide(prs, 103, "A3 · Why attention is order-free (proof sketch)", "Appendix")
    bullets(s, ["Re-order cars 1…N with τ: keys and values are re-ordered the same way, query (our car) unchanged",
                "Each weight α_i depends only on car i's own key → α moves together with its car",
                "z = Σ α_i v_i is a sum: the order of the terms does not matter → z unchanged → Q unchanged",
                "Empty rows: score set to −10⁹ before softmax → weight ≈ 0 → any number of cars works",
                "We test it in code: shuffling the rows gives identical Q-values (torch.allclose)"],
            L, Inches(1.8), CW, Inches(4.5), 18)

    s = new_slide(prs, 104, "A4 · The city journey: details", "Appendix")
    bullets(s, ["Map: the paper's intersection copied 6×, 122 m apart; roads meet exactly (2 × (50 m + 11 m))",
                "Journey: random map-edge start → map-edge finish with ≥ 3 junctions and at least one turn (44 options)",
                "Turn at each junction from the shortest path (breadth-first search in the road graph)",
                "Agent input: positions relative to the junction it approaches, goal = the exit it must take there",
                "Blind-spot fix: keep the junction just crossed in view until halfway to the next one (+3.5 pts)",
                "Tow truck: other cars that crash into each other are removed after 2 s (else they block the road)",
                "Fine-tuning (600 episodes, ε 0.2 → 0.05) made it more aggressive: 44 % vs 52.5 % finished"],
            L, Inches(1.8), CW, Inches(5), 17)

    s = new_slide(prs, 105, "A5 · All results", "Appendix")
    table(s, [["Experiment", "Attention", "FCN/List", "CNN/Grid", "Random"],
              ["Left turn: success / crash %", "67 / 12", "55 / 41", "48 / 17", "20 / 40"],
              ["Dense traffic: success / crash %", "42 / 17", "32 / 61", "27 / 25", "15 / 54"],
              ["12 routes: success / crash %", "70 / 18", "65 / 34", "-", "39 / 28"],
              ["12 routes: left / straight / right success %", "52 / 60 / 98", "41 / 57 / 98", "-", "31 / 24 / 64"],
              ["City journey (zero-shot): finish / crash %", "52.5 / 45.5", "34 / 65.5", "-", "37 / 43"]],
          L, Inches(1.7), CW, [Inches(4.4), Inches(2.0), Inches(2.0), Inches(1.8), Inches(1.9)], size=15,
          row_h=Inches(0.55))
    text(s, "Means over seeds (3 for the paper task, 2 for the extensions). Files: results/*.csv", L, Inches(5.4), CW,
         Inches(0.5), 15, INK_2)

    os.makedirs("slides", exist_ok=True)
    out = "slides/BanditLAB_Final_Presentation.pptx"
    prs.save(out)
    print("->", out)


# ----------------------------------------------------------------------------- printable script
def script_docx():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Segoe UI", DPt(12)
    doc.add_heading("Bandit LAB - presentation script (read aloud)", 0)
    doc.add_paragraph("Social Attention for Autonomous Decision-Making in Dense Traffic (Leurent & Mercat, NeurIPS 2019 "
                      "workshop). 18 slides, about 10 minutes. Each speaker clicks through their own slides. "
                      "Appendix slides A1-A5 are only for questions.")
    total = sum(v[1] for v in SCRIPT.values())
    doc.add_paragraph(f"Planned speaking time: {total // 60} min {total % 60} s.")
    names = {1: "Speaker 1 - slides 1-5: problem and reinforcement learning (DQN)",
             2: "Speaker 2 - slides 6-10: the paper's idea, attention maths and code",
             3: "Speaker 3 - slides 11-18: experiments, results, extensions, demo, conclusion"}
    current = None
    for no in sorted(SCRIPT):
        spk, secs, words = SCRIPT[no]
        if spk != current:
            current = spk
            spk_secs = sum(v[1] for v in SCRIPT.values() if v[0] == spk)
            h = doc.add_heading(f"{names[spk]}  (~{spk_secs // 60} min {spk_secs % 60} s)", 1)
        p = doc.add_paragraph()
        r = p.add_run(f"Slide {no}  (~{secs} s)")
        r.bold = True
        r.font.color.rgb = DRGB(42, 120, 214)
        doc.add_paragraph(words)
    doc.add_heading("If the TA asks (short answers)", 1)
    qa = [("Why DQN and not PPO?", "The paper uses DQN, and our actions are discrete (3 choices) - DQN's natural setting."),
          ("What is Double DQN?", "The online network chooses the next action and the target network scores it. "
                                  "Plain DQN uses one network for both and over-estimates values."),
          ("Why a replay buffer and a target network?", "Replay breaks the correlation between consecutive steps; "
                                                        "the target network keeps the target steady for 512 steps."),
          ("Why gamma 0.95?", "Episodes are short (13 decisions); 0.95 looks about 1/(1-0.95) = 20 steps ahead."),
          ("What does permutation invariant mean?", "Re-ordering the other cars does not change the decision. "
                                                    "Attention is a weighted sum, so the order cannot matter."),
          ("Why divide by the square root of d_h?", "Dot products of long random vectors grow with their length; "
                                                    "dividing by sqrt(32) keeps the softmax from becoming too sharp."),
          ("Why only 3 seeds?", "Each run takes about an hour on a laptop; the paper used 120 seeds on a cluster. "
                                "We show 95% confidence bands."),
          ("How does one agent drive a whole city?", "A route planner picks the turns; at each junction the agent sees "
                                                     "positions relative to that junction, like in training."),
          ("Why did fine-tuning make it worse?", "Every second at full speed earns reward on a 90 s journey, so a short "
                                                 "fine-tune made it more aggressive and it forgot caution."),
          ("Why can't it avoid obstacles?", "It only controls speed and roads have one lane per direction; that needs "
                                            "lane changes - future work.")]
    for q, a in qa:
        p = doc.add_paragraph()
        p.add_run(q + "  ").bold = True
        p.add_run(a)
    out = "slides/BanditLAB_Presentation_Script.docx"
    doc.save(out)
    print("->", out, f"({total} s of speech)")


if __name__ == "__main__":
    build()
    script_docx()
