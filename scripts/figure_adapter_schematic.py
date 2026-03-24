"""
Figure 3 — Adapter architecture schematic.

Shows a single camera image flowing through:
  1. ResNet-101 backbone + FPN  (frozen, grey)
  2. Residual injection point   (red circle ⊕)
  3. BEV Encoder               (frozen, grey)
  4. Detection head             (frozen, grey)

DAv2 branch (top):
  Camera image → DAv2 ViT-S (frozen) → channel select (96/384) →
  Conv 3×3 (trainable) → ReLU → Conv 1×1 (trainable) → ×α → ⊕

Colour coding:
  Dark grey  = frozen BEVFormer components
  Warm red   = trainable adapter components
  Blue       = DAv2 (frozen)

Output: E:\bev_research\figures\adapter_schematic.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

BG         = "#FAFAFA"
C_FROZEN   = "#4A5568"   # frozen blocks
C_ADAPTER  = "#C53030"   # trainable adapter
C_DAV2     = "#2B6CB0"   # frozen DAv2
C_ARROW    = "#2D3748"
C_ANNOT    = "#555555"
C_WHITE    = "#FFFFFF"
C_INJECT   = "#D69E2E"   # injection point

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


def box(ax, x, y, w, h, color, label, sub=None,
        fs=9, sub_fs=7.5, alpha=0.93, zorder=3, text_color=C_WHITE):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.07",
                       linewidth=1.3,
                       edgecolor=color,
                       facecolor=color,
                       alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    cx, cy = x + w / 2, y + h / 2
    dy = 0.15 if sub else 0
    ax.text(cx, cy + dy, label, ha="center", va="center",
            color=text_color, fontsize=fs, fontweight="bold", zorder=zorder + 1)
    if sub:
        ax.text(cx, cy - 0.28, sub, ha="center", va="center",
                color=text_color, fontsize=sub_fs, zorder=zorder + 1, style="italic")


def arr(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.7, label=None,
        fs=8, lo=(0, 0.15), rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=14,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=6)
    if label:
        mx = (x0 + x1) / 2 + lo[0]
        my = (y0 + y1) / 2 + lo[1]
        ax.text(mx, my, label, ha="center", fontsize=fs, color=C_ANNOT, zorder=7)


def plus_circle(ax, cx, cy, r=0.28, color=C_INJECT, zorder=8):
    circ = plt.Circle((cx, cy), r, color=color, zorder=zorder)
    ax.add_patch(circ)
    ax.text(cx, cy, "⊕", ha="center", va="center",
            color="white", fontsize=14, fontweight="bold", zorder=zorder + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Camera input (leftmost)
# ─────────────────────────────────────────────────────────────────────────────
box(ax, 0.2, 3.7, 1.5, 0.9, C_FROZEN, "Camera\nImages",
    sub="6-cam, 1600×900", fs=8.5, sub_fs=7)

# ─────────────────────────────────────────────────────────────────────────────
# BEVFormer main path (horizontal, y ≈ 4.1)
# ─────────────────────────────────────────────────────────────────────────────

arr(ax, 1.7, 4.15, 2.5, 4.15)

# ResNet-101 + FPN
box(ax, 2.5, 3.7, 2.2, 0.9, C_FROZEN, "ResNet-101", sub="+ FPN neck", fs=9)

arr(ax, 4.7, 4.15, 5.5, 4.15, label="img_feat\n256-D, 4 levels", lo=(0, 0.32), fs=7.5)

# Injection point
plus_circle(ax, 5.9, 4.15)

arr(ax, 6.18, 4.15, 7.1, 4.15)

# BEV Encoder
box(ax, 7.1, 3.7, 2.2, 0.9, C_FROZEN, "BEV Encoder", sub="spatial cross-attn", fs=9)

arr(ax, 9.3, 4.15, 10.1, 4.15, label="bev_embed", lo=(0, 0.25), fs=7.5)

# Detection Head
box(ax, 10.1, 3.7, 2.1, 0.9, C_FROZEN, "Det. Head", sub="DETR-style", fs=9)

arr(ax, 12.2, 4.15, 13.1, 4.15)

# Output
box(ax, 13.1, 3.7, 2.2, 0.9, "#276749", "Predictions",
    sub="mAP / NDS", fs=9, alpha=0.88)

# Frozen label under main path
ax.text(7.0, 3.35, "← All BEVFormer components frozen (no gradient) →",
        ha="center", fontsize=8, color="#777777", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# DAv2 branch (top, y ≈ 6.5–7.8)
# ─────────────────────────────────────────────────────────────────────────────

# Arrow: camera images → DAv2
arr(ax, 0.95, 4.6, 0.95, 7.05, color=C_DAV2, lw=1.5, rad=0.0)

# DAv2 ViT-S
box(ax, 0.2, 7.05, 2.0, 0.85, C_DAV2, "DAv2 ViT-S", sub="frozen, 384-D", fs=9)

arr(ax, 2.2, 7.47, 3.1, 7.47, label="(B·N_cam, 384, H/14, W/14)", lo=(0, 0.28), fs=7)

# Channel select box
box(ax, 3.1, 7.05, 2.2, 0.85, "#5A67D8", "Channel Select",
    sub="96 / 384 depth-scale\ninvariant (E6 only)", fs=8.5, sub_fs=6.8)

arr(ax, 5.3, 7.47, 6.2, 7.47, label="(B·N_cam, 96|384, H', W')", lo=(0, 0.28), fs=7)

# Conv 3×3 (trainable)
box(ax, 6.2, 7.05, 1.9, 0.85, C_ADAPTER, "Conv 3×3", sub="trainable, 256-D", fs=8.5)

arr(ax, 8.1, 7.47, 8.7, 7.47)

# ReLU
box(ax, 8.7, 7.12, 0.9, 0.7, "#744210", "ReLU", fs=8.5)

arr(ax, 9.6, 7.47, 10.3, 7.47)

# Conv 1×1 (trainable)
box(ax, 10.3, 7.05, 1.9, 0.85, C_ADAPTER, "Conv 1×1", sub="trainable, 256-D", fs=8.5)

arr(ax, 12.2, 7.47, 12.9, 7.47, label="δ (delta)", lo=(0, 0.24), fs=7.5)

# ×α scale box
box(ax, 12.9, 7.12, 0.9, 0.7, "#744210", "× α", sub="0.1", fs=9)

# Arrow down from ×α to injection point
arr(ax, 13.35, 7.12, 13.35, 5.8, color=C_INJECT, lw=1.5)
arr(ax, 13.35, 5.8, 5.9, 4.43, color=C_INJECT, lw=1.5,
    label="α · δ injected\nat FPN level 0", lo=(-2.5, 0.25), fs=7.5)

# ─────────────────────────────────────────────────────────────────────────────
# Parameter count annotation
# ─────────────────────────────────────────────────────────────────────────────
ax.text(7.8, 6.5,
        "E3-A/B/C (384-ch): Conv(384→256) + Conv(256→256) = 164 K params\n"
        "E6        (96-ch):  Conv(96→256) + Conv(256→256) =  91 K params",
        ha="center", fontsize=8, color=C_ANNOT,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9E6",
                  edgecolor="#D69E2E", lw=1.0),
        zorder=9)

# ─────────────────────────────────────────────────────────────────────────────
# Measurement annotations (hooks)
# ─────────────────────────────────────────────────────────────────────────────
ax.annotate("", xy=(3.6, 4.6), xytext=(3.6, 5.5),
            arrowprops=dict(arrowstyle="-|>", color=C_ANNOT, lw=1.2,
                            linestyle="dashed"), zorder=5)
ax.text(3.6, 5.65, "hook ①\nimg_feat", ha="center", fontsize=7.5,
        color=C_ANNOT, style="italic")

ax.annotate("", xy=(8.2, 4.6), xytext=(8.2, 5.5),
            arrowprops=dict(arrowstyle="-|>", color=C_ANNOT, lw=1.2,
                            linestyle="dashed"), zorder=5)
ax.text(8.2, 5.65, "hook ②\nbev_embed", ha="center", fontsize=7.5,
        color=C_ANNOT, style="italic")

ax.text(5.9, 5.8,
        "CKA (debiased): 0.003 → 0.009\nCosine sim.: 0.424 → 0.890 (81%)",
        ha="center", fontsize=7.5, color=C_ANNOT,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#EFF8FF",
                  edgecolor="#63B3ED", lw=1.0), zorder=9)

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_FROZEN,  label="Frozen (no grad)"),
    mpatches.Patch(color=C_DAV2,    label="Frozen DAv2 ViT-S"),
    mpatches.Patch(color=C_ADAPTER, label="Trainable adapter"),
    mpatches.Patch(color=C_INJECT,  label="Residual injection ⊕"),
]
ax.legend(handles=legend_items, loc="lower right",
          fontsize=8.5, framealpha=0.9, edgecolor="#CCCCCC",
          bbox_to_anchor=(1.0, 0.0))

# Title
ax.text(8.0, 8.65, "Adapter Architecture: DAv2 Feature Injection into BEVFormer",
        ha="center", fontsize=13, fontweight="bold", color="#1A202C")

plt.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    out = rf"E:\bev_research\figures\adapter_schematic.{ext}"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}", flush=True)
plt.close()
