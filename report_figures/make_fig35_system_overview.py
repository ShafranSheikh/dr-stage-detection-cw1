"""
Figure 35 - system overview for the CW1 report.

Draws the whole project as one vertical pipeline: the dataset, the four
Kaggle notebooks and the Streamlit prototype, with the leakage controls
listed underneath.

Nothing is computed here. Every number is copied from the results the
notebooks actually produced, so this file is only a drawing script.

Run:  python make_fig35_system_overview.py
Out:  fig35_system_overview.png  (written next to this script)
"""

import os
import matplotlib
matplotlib.use("Agg")                      # write a file, do not open a window
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------- palette
# The same colours as figure 36, so the two diagrams read as one set.
PAGE_BG    = "#fafafa"
INK        = "#1a1a1a"          # headings
INK_SOFT   = "#4a4a4a"          # body text
GREY_FACE,   GREY_EDGE   = "#efeeea", "#8a8a85"     # the raw dataset
BLUE_FACE,   BLUE_EDGE   = "#d9e8fa", "#1a73c8"     # code we wrote
ORANGE_FACE, ORANGE_EDGE = "#fbe6d8", "#e2571f"     # the part a user touches
LINE       = "#9a9a95"          # arrows and rules
PANEL_FACE, PANEL_EDGE   = "#f2f1ee", "#cfcdc8"     # the footer panel

# ------------------------------------------------------------ the content
# (name, what it does, output line 1, output line 2, colour role)
STAGES = [
    ("APTOS 2019", "Kaggle dataset",
     "3,662 fundus photographs, one grade per image",
     "stages 0-4   ·   17 different image sizes",
     "grey"),
    ("Notebook 1", "explore & split",
     "duplicate check drops 173 images, 3,489 kept",
     "stratified 70/15/15  →  2,441 / 523 / 525",
     "blue"),
    ("Notebook 2", "preprocess",
     "retina crop  →  common window  →  512 px",
     "variants A / B / C, camera-only baseline",
     "blue"),
    ("Notebook 3", "train & experiment",
     "E1-E5: 15 runs across two seeds",
     "EfficientNetB3 + CLAHE wins  →  best_model.keras",
     "blue"),
    ("Notebook 4", "evaluate once",
     "525 test images, opened one time only",
     "metrics, error analysis, Grad-CAM  →  figures 26-34",
     "blue"),
    ("Prototype", "Streamlit, on the Mac",
     "the trained model + the same preprocessing code",
     "upload  →  answer  →  Grad-CAM heatmap",
     "orange"),
]

# six short rules, printed as two columns of three
LEAKAGE = [
    "duplicate groups split together",
    "augmentation only on the training split",
    "one saved split used by every notebook",
    "oversampling only on the training split",
    "preprocessing uses one image at a time",
    "test split scored once, at the very end",
]

FACES = {"grey":   (GREY_FACE, GREY_EDGE),
         "blue":   (BLUE_FACE, BLUE_EDGE),
         "orange": (ORANGE_FACE, ORANGE_EDGE)}

# ------------------------------------------------------------ the drawing
fig, ax = plt.subplots(figsize=(11.0, 8.4), dpi=170)
fig.patch.set_facecolor(PAGE_BG)
ax.set_facecolor(PAGE_BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# geometry, in the 0-100 space set above
X0, X1 = 3.0, 88.0          # left and right edge of every stage box
BAND_H = 8.2                # height of one stage box
GAP    = 3.2                # blank space between boxes, holds the arrow
TOP    = 89.5               # top edge of the first box
NAME_X = X0 + 3.0           # where the bold stage name starts
TEXT_X = X0 + 27.0          # where the two output lines start
PAD_R  = 2.0                # text must stop this far from the right edge

ax.text(50, 96.0, "Figure 35 - The system, from dataset to prototype",
        ha="center", va="center", fontsize=17, fontweight="bold", color=INK)

band_tops, band_bottoms = [], []
checks = []                 # (label, text object, x limit) for the fit test

for i, (name, does, line1, line2, role) in enumerate(STAGES):
    top = TOP - i * (BAND_H + GAP)
    bottom = top - BAND_H
    band_tops.append(top)
    band_bottoms.append(bottom)
    face, edge = FACES[role]

    ax.add_patch(FancyBboxPatch(
        (X0, bottom), X1 - X0, BAND_H,
        boxstyle="round,pad=0,rounding_size=1.5",
        facecolor=face, edgecolor=edge, linewidth=1.8, zorder=2))

    mid = (top + bottom) / 2
    # left column - what this stage is
    ax.text(NAME_X, mid + 1.5, name, ha="left", va="center",
            fontsize=13.5, fontweight="bold", color=INK, zorder=3)
    ax.text(NAME_X, mid - 1.9, does, ha="left", va="center",
            fontsize=11.0, color=INK_SOFT, zorder=3)
    # right column - what it produced
    for dy, txt in ((1.6, line1), (-1.9, line2)):
        t = ax.text(TEXT_X, mid + dy, txt, ha="left", va="center",
                    fontsize=11.0, color=INK_SOFT, zorder=3)
        checks.append((name + ": " + txt, t, X1 - PAD_R))

    # arrow down to the next stage
    if i < len(STAGES) - 1:
        ax.annotate("", xy=(50, bottom - GAP + 0.5), xytext=(50, bottom - 0.3),
                    arrowprops=dict(arrowstyle="-|>", color=LINE,
                                    linewidth=1.5, mutation_scale=13),
                    zorder=1)

# The one link that is not a straight hand-off: the prototype imports
# notebook 2's preprocessing file, so the app sees what the model was
# trained on. Drawn in the right-hand gutter so it crosses nothing.
GUTTER = X1 + 3.2
nb2_mid   = (band_tops[2] + band_bottoms[2]) / 2
proto_mid = (band_tops[5] + band_bottoms[5]) / 2
dashes = dict(color=LINE, linewidth=1.3, linestyle=(0, (4, 3)), zorder=1)
ax.plot([X1, GUTTER], [nb2_mid, nb2_mid], **dashes)
ax.plot([GUTTER, GUTTER], [nb2_mid, proto_mid], **dashes)
ax.annotate("", xy=(X1 + 0.3, proto_mid), xytext=(GUTTER, proto_mid),
            arrowprops=dict(arrowstyle="-|>", color=LINE, linewidth=1.3,
                            linestyle=(0, (4, 3)), mutation_scale=13),
            zorder=1)
ax.text(GUTTER + 2.4, (nb2_mid + proto_mid) / 2, "same preprocessing file",
        ha="center", va="center", fontsize=9.5, color=INK_SOFT, rotation=90)

# ------------------------------------------------------- leakage controls
panel_top    = band_bottoms[-1] - 4.5
panel_bottom = 0.5
ax.add_patch(FancyBboxPatch(
    (X0, panel_bottom), X1 - X0, panel_top - panel_bottom,
    boxstyle="round,pad=0,rounding_size=1.3",
    facecolor=PANEL_FACE, edgecolor=PANEL_EDGE, linewidth=1.4, zorder=2))

ax.text(NAME_X, panel_top - 3.4, "Leakage controls, applied throughout",
        ha="left", va="center", fontsize=12.0, fontweight="bold",
        color=INK, zorder=3)

col_x = [NAME_X, X0 + 44.0]                       # two columns
row_y = [panel_top - 7.8, panel_top - 12.0, panel_top - 16.2]
for n, item in enumerate(LEAKAGE):
    x = col_x[n % 2]
    t = ax.text(x, row_y[n // 2], "·   " + item, ha="left", va="center",
                fontsize=10.5, color=INK_SOFT, zorder=3)
    limit = (col_x[1] - 1.5) if n % 2 == 0 else (X1 - PAD_R)
    checks.append(("leakage: " + item, t, limit))

# ------------------------------------------ does every line fit its box?
# Renders once, then compares each text's right edge against its limit,
# so a wording change can never silently overflow again.
fig.canvas.draw()
inv = ax.transData.inverted()
overflow = False
for label, t, limit in checks:
    right = inv.transform(t.get_window_extent())[1][0]
    if right > limit:
        overflow = True
        print("OVERFLOW by %.1f : %s" % (right - limit, label))
print("fit check:", "FAILED" if overflow else "all lines fit")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fig35_system_overview.png")
fig.savefig(out, dpi=170, facecolor=PAGE_BG, bbox_inches="tight",
            pad_inches=0.25)
print("wrote", out)
