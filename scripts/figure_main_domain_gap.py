"""
figure_main_domain_gap.py

Generates Figure 1 for the BEV domain-gap paper.

Layout  (10 × 3.6 inches, 300 DPI):
  ┌──────────────────┬──────────────────┬────────────────────┐
  │  Boston CAM_FRONT│ Singapore        │  Error-component   │
  │  + GT boxes      │ CAM_FRONT        │  gap bar chart     │
  │  (source city)   │ + GT boxes       │  mATE  mASE  mAOE  │
  │                  │ (target city)    │  mAVE  mAAE        │
  └──────────────────┴──────────────────┴────────────────────┘

Usage:
    conda activate bev310
    python E:\\bev_research\\scripts\\figure_main_domain_gap.py

Output:
    E:\\bev_research\\figures\\figure1_domain_gap.pdf
    E:\\bev_research\\figures\\figure1_domain_gap.png  (300 DPI)
"""

import pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
DATAROOT   = Path(r"E:\datasets\nuscenes")
BOSTON_PKL = DATAROOT / "nuscenes_infos_temporal_val_boston.pkl"
SING_PKL   = DATAROOT / "nuscenes_infos_temporal_val_singapore.pkl"
OUT_DIR    = Path(r"E:\bev_research\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Frame selection ──────────────────────────────────────────────────────────
# Change these indices to pick different representative frames.
# Current selection: day-time, ~12–15 GT boxes visible in front camera.
BOSTON_FRAME_IDX    = 2    # n008-2018-08-01 (Boston-Seaport, daytime)
SINGAPORE_FRAME_IDX = 90   # n015-2018-07-18 (Singapore-OneNorth, daytime)

# ─── Measured performance numbers (from paper Table 1) ────────────────────────
BOSTON_METRICS    = dict(mAP=0.425, mATE=0.666, mASE=0.280, mAOE=0.321, mAVE=0.461, mAAE=0.157)
SINGAPORE_METRICS = dict(mAP=0.367, mATE=0.726, mASE=0.354, mAOE=0.472, mAVE=0.615, mAAE=0.353)

# Percentage increase per TP metric (Singapore relative to Boston)
GAP_PCT = dict(mATE=9, mASE=26, mAOE=47, mAVE=33, mAAE=125)

# ─── nuScenes class colour palette ────────────────────────────────────────────
CLASS_COLORS = {
    "car":                  "#4E9AF1",   # sky blue
    "truck":                "#1A5FA8",   # dark blue
    "bus":                  "#0D2F6B",   # navy
    "trailer":              "#7B3FB5",   # purple
    "construction_vehicle": "#8B4513",   # saddle brown
    "pedestrian":           "#E84040",   # red
    "motorcycle":           "#F5A623",   # amber
    "bicycle":              "#4CAF50",   # green
    "barrier":              "#78909C",   # blue-grey
    "traffic_cone":         "#FFC107",   # yellow
}
DEFAULT_COLOR = "#AAAAAA"

NUSCENES_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]


# ─── 3D box → 2D projection helpers ──────────────────────────────────────────

def box_corners_lidar(cx, cy, cz, l, w, h, yaw):
    """Return 8 corners of a 3-D box in LiDAR frame (x-forward, y-left, z-up)."""
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    dx, dy, dz = l / 2, w / 2, h / 2
    # 8 corners in box-local frame (columns: ±x, ±y, ±z)
    local = np.array([
        [ dx,  dy,  dz],  # 0: front-right-top
        [ dx, -dy,  dz],  # 1: front-left-top
        [-dx,  dy,  dz],  # 2: back-right-top
        [-dx, -dy,  dz],  # 3: back-left-top
        [ dx,  dy, -dz],  # 4: front-right-bot
        [ dx, -dy, -dz],  # 5: front-left-bot
        [-dx,  dy, -dz],  # 6: back-right-bot
        [-dx, -dy, -dz],  # 7: back-left-bot
    ])
    R = np.array([[cos_y, -sin_y, 0],
                  [sin_y,  cos_y, 0],
                  [0,      0,     1]])
    return (R @ local.T).T + np.array([cx, cy, cz])   # (8, 3)


# Edges connecting the 8 corners
BOX_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 3),   # top face
    (4, 5), (4, 6), (5, 7), (6, 7),   # bottom face
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]


def project_corners(corners_lidar, R_s2l, t_s2l, K):
    """
    Project 8 LiDAR-frame corners onto a camera image.

    sensor2lidar_rotation (R_s2l) and sensor2lidar_translation (t_s2l) give
    the transform FROM camera-sensor coordinates TO LiDAR coordinates:
        p_lidar = R_s2l @ p_sensor + t_s2l

    Inverse (LiDAR → sensor):
        p_sensor = R_s2l.T @ (p_lidar - t_s2l)
    """
    R_l2s = R_s2l.T
    t_l2s = -R_l2s @ t_s2l
    corners_cam = (R_l2s @ corners_lidar.T).T + t_l2s   # (8, 3)

    # Only project corners in front of the camera
    in_front = corners_cam[:, 2] > 0.1
    if not np.any(in_front):
        return None, None

    # Homogeneous projection: K @ p_cam → (u, v, w)
    uvw = (K @ corners_cam.T).T   # (8, 3)
    uv  = uvw[:, :2] / uvw[:, 2:3]   # (8, 2) pixel coordinates
    return uv, in_front


def draw_boxes_on_image(ax, info, cam_key="CAM_FRONT"):
    """Load image and draw projected GT boxes coloured by class."""
    cam = info["cams"][cam_key]
    img_path = cam["data_path"]
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    R_s2l = np.array(cam["sensor2lidar_rotation"])
    t_s2l = np.array(cam["sensor2lidar_translation"])
    K     = np.array(cam["cam_intrinsic"])

    H, W = img_rgb.shape[:2]

    ax.imshow(img_rgb)

    gt_boxes  = info.get("gt_boxes",  np.zeros((0, 7)))
    gt_names  = info.get("gt_names",  np.array([]))
    if len(gt_boxes) == 0:
        return

    for box, name in zip(gt_boxes, gt_names):
        cx, cy, cz, l, w, h, yaw = box
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)

        corners = box_corners_lidar(cx, cy, cz, l, w, h, yaw)
        uv, in_front = project_corners(corners, R_s2l, t_s2l, K)
        if uv is None:
            continue

        # Draw each edge; skip if both endpoints are behind camera or off-screen
        for i0, i1 in BOX_EDGES:
            if not (in_front[i0] and in_front[i1]):
                continue
            x0, y0 = uv[i0]
            x1, y1 = uv[i1]
            # Clip to image bounds with a generous margin
            if (max(x0, x1) < -50 or min(x0, x1) > W + 50 or
                    max(y0, y1) < -50 or min(y0, y1) > H + 50):
                continue
            ax.plot([x0, x1], [y0, y1], color=color,
                    linewidth=1.4, alpha=0.92, solid_capstyle="round")

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")


# ─── Bar chart helpers ────────────────────────────────────────────────────────

METRIC_LABELS = ["mATE", "mASE", "mAOE", "mAVE", "mAAE"]
METRIC_UNITS  = ["m", "—", "rad", "m/s", "—"]


def draw_error_chart(ax):
    """Grouped bar chart: Boston vs Singapore per TP error metric."""
    boston_vals = [BOSTON_METRICS[m] for m in METRIC_LABELS]
    sing_vals   = [SINGAPORE_METRICS[m] for m in METRIC_LABELS]
    x = np.arange(len(METRIC_LABELS))
    bar_w = 0.34

    # Palette: Boston – muted steel blue; Singapore – warm coral
    C_BOS  = "#4477AA"
    C_SING = "#CC6677"

    bars_b = ax.bar(x - bar_w / 2, boston_vals, bar_w,
                    color=C_BOS,  label="Boston (source)",
                    zorder=3, linewidth=0.6, edgecolor="white")
    bars_s = ax.bar(x + bar_w / 2, sing_vals, bar_w,
                    color=C_SING, label="Singapore (target)",
                    zorder=3, linewidth=0.6, edgecolor="white")

    # Annotate % increase above each Singapore bar
    for i, (bv, sv) in enumerate(zip(boston_vals, sing_vals)):
        pct = GAP_PCT[METRIC_LABELS[i]]
        color = "#BB0000" if pct >= 40 else "#884400"
        ax.text(x[i] + bar_w / 2, sv + 0.010,
                f"+{pct}%",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=color, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=9)
    ax.set_ylabel("Error value", fontsize=9)
    ax.set_ylim(0, max(sing_vals) * 1.28)
    ax.set_xlim(-0.6, len(METRIC_LABELS) - 0.4)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(fontsize=8, framealpha=0.9, edgecolor="none",
                       loc="upper left", bbox_to_anchor=(0.01, 0.99))

    # Small note about what the annotation means
    ax.text(0.99, 0.01, "% = Singapore increase\nrelative to Boston",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="#555555", style="italic")


# ─── City info overlay ────────────────────────────────────────────────────────

def city_overlay(ax, city_name, map_str, color, n_boxes, alpha=0.82):
    """Add a semi-transparent label block on the camera panel."""
    props = dict(boxstyle="round,pad=0.35", facecolor=color,
                 alpha=alpha, linewidth=0)
    ax.text(0.015, 0.975, city_name,
            transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            color="white", va="top", ha="left",
            bbox=props, zorder=10)
    # mAP & box count in smaller text at bottom
    ax.text(0.015, 0.028,
            f"mAP = {map_str}   |   {n_boxes} GT objects",
            transform=ax.transAxes, fontsize=7.5,
            color="white", va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#111111",
                      alpha=0.60, linewidth=0),
            zorder=10)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading PKL files...")
    with open(BOSTON_PKL, "rb") as f:
        db = pickle.load(f)
    with open(SING_PKL, "rb") as f:
        ds = pickle.load(f)

    boston_infos = db.get("infos", db.get("data_list", db))
    sing_infos   = ds.get("infos", ds.get("data_list", ds))

    info_b = boston_infos[BOSTON_FRAME_IDX]
    info_s = sing_infos[SINGAPORE_FRAME_IDX]

    n_boxes_b = len(info_b.get("gt_boxes", []))
    n_boxes_s = len(info_s.get("gt_boxes", []))

    print(f"Boston  frame {BOSTON_FRAME_IDX}: {n_boxes_b} GT boxes")
    print(f"Singapore frame {SINGAPORE_FRAME_IDX}: {n_boxes_s} GT boxes")

    # ── Figure layout ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "axes.linewidth":   0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi":       300,
    })

    fig = plt.figure(figsize=(10.2, 3.6))
    gs  = GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[2.05, 2.05, 1.9],
        wspace=0.06,
        left=0.01, right=0.99,
        top=0.97,  bottom=0.05,
    )

    ax_b   = fig.add_subplot(gs[0])
    ax_s   = fig.add_subplot(gs[1])
    ax_bar = fig.add_subplot(gs[2])

    # ── Boston camera panel ───────────────────────────────────────────────────
    print("Rendering Boston frame...")
    draw_boxes_on_image(ax_b, info_b, cam_key="CAM_FRONT")
    city_overlay(ax_b, "Boston  (source)",
                 f"{BOSTON_METRICS['mAP']:.3f}", "#2255AA", n_boxes_b)

    # ── Singapore camera panel ────────────────────────────────────────────────
    print("Rendering Singapore frame...")
    draw_boxes_on_image(ax_s, info_s, cam_key="CAM_FRONT")
    city_overlay(ax_s, "Singapore  (target)",
                 f"{SINGAPORE_METRICS['mAP']:.3f}", "#AA2222", n_boxes_s)

    # ── Error bar chart ───────────────────────────────────────────────────────
    ax_bar.set_position([0.70, 0.13, 0.285, 0.82])   # fine-tune right panel
    draw_error_chart(ax_bar)

    # ── Thin divider between image panels and chart ───────────────────────────
    fig.add_artist(plt.Line2D(
        [0.69, 0.69], [0.04, 0.98],
        transform=fig.transFigure, color="#CCCCCC", linewidth=0.8,
    ))

    # ── Save ──────────────────────────────────────────────────────────────────
    pdf_path = OUT_DIR / "figure1_domain_gap.pdf"
    png_path = OUT_DIR / "figure1_domain_gap.png"

    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)

    print(f"\nSaved:\n  {pdf_path}\n  {png_path}")
    print("\nTo use different frames, edit BOSTON_FRAME_IDX and SINGAPORE_FRAME_IDX")
    print("at the top of the script and re-run.")


if __name__ == "__main__":
    main()
