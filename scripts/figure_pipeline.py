"""
Figure 2 — Experimental pipeline overview diagram.

Layout (left → right):
  [nuScenes] → [City Split] → [BEVFormer-Base] → [Hooks: img_feat / bev_embed]
                                    ↓
                           [DAv2 ViT-S (frozen)]
                                    ↓
                         [Adapter (5 configs E3→E5)]
                                    ↓
                        [Singapore eval: mAP / NDS]

Annotations on arrows and boxes explain:
  - N_boston=3090, N_singapore=2929
  - Hooks at FPN output and BEV encoder output
  - CKA + cosine similarity measurement
  - Five adapter failure modes

Output: E:\bev_research\figures\pipeline_diagram.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
C_DATA    = "#2D6A9F"   # dataset / data boxes
C_MODEL   = "#1E6B52"   # model boxes
C_FROZEN  = "#5B5B5B"   # frozen components
C_TRAIN   = "#B85C38"   # trainable components
C_METRIC  = "#7B4FA0"   # measurement / metric boxes
C_ARROW   = "#444444"
C_ANNOT   = "#333333"
BG        = "#F8F8F8"

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


def rounded_box(ax, x, y, w, h, color, label, sublabel=None,
                fontsize=10, sublabel_fs=8, alpha=0.92, zorder=3):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        linewidth=1.4,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(box)
    mid_x = x + w / 2
    mid_y = y + h / 2
    offset = 0.15 if sublabel else 0
    ax.text(mid_x, mid_y + offset, label,
            ha="center", va="center", color="white",
            fontsize=fontsize, fontweight="bold", zorder=zorder + 1)
    if sublabel:
        ax.text(mid_x, mid_y - 0.28, sublabel,
                ha="center", va="center", color="white",
                fontsize=sublabel_fs, zorder=zorder + 1, style="italic")


def arrow(ax, x0, y0, x1, y1, label=None, color=C_ARROW, lw=1.8,
          fs=8, annot_offset=(0, 0.18)):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=lw,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )
    if label:
        mx = (x0 + x1) / 2 + annot_offset[0]
        my = (y0 + y1) / 2 + annot_offset[1]
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=fs, color=C_ANNOT, zorder=6)


# ─────────────────────────────────────────────────────────────────────────────
# Row 1 (y ≈ 5.8):  nuScenes  →  City Split  →  Boston val / Singapore val
# ─────────────────────────────────────────────────────────────────────────────

# nuScenes box
rounded_box(ax, 0.3, 5.6, 2.0, 1.1, C_DATA, "nuScenes", "700 training scenes")

arrow(ax, 2.3, 6.15, 3.0, 6.15)

# City split
rounded_box(ax, 3.0, 5.6, 2.2, 1.1, C_DATA, "City Split", "by location field")

# Boston arrow
arrow(ax, 5.2, 6.6, 6.2, 7.0, label="Boston\n3,090 frames", annot_offset=(-0.25, 0.05))
# Singapore arrow
arrow(ax, 5.2, 5.7, 6.2, 5.4, label="Singapore\n2,929 frames", annot_offset=(0.3, 0.05))

# Boston val box
rounded_box(ax, 6.2, 6.7, 2.0, 0.85, C_DATA, "Boston val", "source domain", sublabel_fs=7)

# Singapore val box
rounded_box(ax, 6.2, 5.1, 2.0, 0.85, C_DATA, "Singapore val", "target domain", sublabel_fs=7)

# ─────────────────────────────────────────────────────────────────────────────
# Centre column (x ≈ 9):  BEVFormer-Base
# ─────────────────────────────────────────────────────────────────────────────

# Arrow from Boston → BEVFormer
arrow(ax, 8.2, 7.12, 9.0, 7.0, color=C_FROZEN)
# Arrow from Singapore → BEVFormer
arrow(ax, 8.2, 5.52, 9.0, 5.7, color=C_FROZEN)

# BEVFormer main box
rounded_box(ax, 9.0, 5.2, 3.0, 2.2, C_MODEL, "BEVFormer-Base", "ResNet-101 + FPN + BEV Encoder",
            fontsize=11, alpha=0.88)

# Hook annotation: img_feat
ax.text(9.1, 6.65, "hook ①", fontsize=7.5, color="white", zorder=7, style="italic")
ax.text(9.1, 6.45, "img_feat", fontsize=7, color="#AAFFCC", zorder=7, fontweight="bold")
ax.text(9.1, 6.28, "(after FPN)", fontsize=6.5, color="#CCCCCC", zorder=7)

# Hook annotation: bev_embed
ax.text(9.1, 5.7, "hook ②", fontsize=7.5, color="white", zorder=7, style="italic")
ax.text(9.1, 5.50, "bev_embed", fontsize=7, color="#AAFFCC", zorder=7, fontweight="bold")
ax.text(9.1, 5.33, "(after BEV enc.)", fontsize=6.5, color="#CCCCCC", zorder=7)

# ─────────────────────────────────────────────────────────────────────────────
# Measurement branch (top, y ≈ 8):  hooks → CKA / cosine analysis
# ─────────────────────────────────────────────────────────────────────────────
rounded_box(ax, 9.0, 3.5, 3.0, 1.35, C_METRIC,
            "Representation Analysis",
            "N=500 density-matched pairs",
            fontsize=9.5, sublabel_fs=7.5, alpha=0.90)

# Arrows down from BEVFormer to analysis
arrow(ax, 10.5, 5.2, 10.5, 4.85, color=C_METRIC, lw=1.5)

ax.text(9.15, 4.55, "Cosine sim. : 0.424→0.890 (81%)", fontsize=7.5,
        color="white", zorder=8, fontweight="bold")
ax.text(9.15, 4.25, "Debiased CKA: 0.003→0.009 (0.5%)", fontsize=7.5,
        color="white", zorder=8, fontweight="bold")
ax.text(9.15, 3.95, "t-SNE drift ratio: 1.30", fontsize=7.5,
        color="white", zorder=8)

# ─────────────────────────────────────────────────────────────────────────────
# DAv2 column (x ≈ 13):  frozen ViT-S → channel selection → adapter
# ─────────────────────────────────────────────────────────────────────────────

rounded_box(ax, 13.0, 5.8, 2.5, 1.2, C_FROZEN,
            "DAv2 ViT-S", "frozen, 384-D features",
            fontsize=9, sublabel_fs=7.5)

ax.text(13.5, 5.55, "96 depth-scale\ninvariant channels\n(Cohen's d=0.09)",
        fontsize=7, ha="center", color=C_ANNOT)

rounded_box(ax, 13.0, 3.8, 2.5, 1.3, C_TRAIN,
            "Adapter", "Conv3×3 + ReLU + Conv1×1\n164 K params",
            fontsize=9, sublabel_fs=7)

# Arrow from DAv2 → Adapter
arrow(ax, 14.25, 5.8, 14.25, 5.1, label="select channels", color=C_FROZEN,
      annot_offset=(0.55, 0.04), fs=7)

# Arrow from Adapter → BEVFormer (injection)
arrow(ax, 13.0, 4.45, 12.0, 5.6, label="residual inject\n(FPN level 0)",
      color=C_TRAIN, lw=1.5, annot_offset=(0.1, 0.1), fs=7)

# ─────────────────────────────────────────────────────────────────────────────
# Bottom row: five experiment configs → Singapore eval → result
# ─────────────────────────────────────────────────────────────────────────────

exp_labels = [
    ("E3-A", "α=0.01\nGrad→0"),
    ("E3-B", "α=0.1\nZero-out"),
    ("E3-C", "α=0.1+cons.\nWrong aug."),
    ("E4",   "Partial\nunfreeze"),
    ("E5",   "Pseudo-\nlabels"),
    ("E6",   "96-ch.\nscale-only"),
]
x_start = 0.4
for i, (tag, desc) in enumerate(exp_labels):
    xb = x_start + i * 2.55
    col = C_TRAIN if tag not in ("E4",) else "#A03070"
    rounded_box(ax, xb, 1.3, 2.1, 1.3, col, tag, desc,
                fontsize=9, sublabel_fs=7, alpha=0.82)

# Arrow from bottom row to Singapore eval note
ax.annotate("", xy=(8.0, 0.8), xytext=(8.0, 1.3),
            arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.5), zorder=5)
ax.text(8.0, 0.5, "All five designs fail for distinct, principled reasons",
        ha="center", fontsize=9, color=C_ANNOT, style="italic")
ax.text(8.0, 0.18, "Baseline: mAP 0.367 (Singapore) — best adapter: mAP 0.360 (E5, epoch 1)",
        ha="center", fontsize=8, color="#884444")

# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
ax.text(8.0, 7.7, "Experimental Pipeline: BEV Domain Gap Diagnosis",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#222222")

plt.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    out = rf"E:\bev_research\figures\pipeline_diagram.{ext}"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}", flush=True)
plt.close()
