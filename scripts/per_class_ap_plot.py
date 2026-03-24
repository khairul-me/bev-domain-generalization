"""
per_class_ap_plot.py
Extract per-class AP from Boston and Singapore eval logs,
produce a comparison bar chart and a ranked degradation table.

FIX vs. guide original:
  Replaced the two regex patterns that assumed "car AP: 0.5731" format.
  The actual MMDetection3D log puts everything on one long line as:
    NuScenes metric/pred_instances_3d_NuScenes/car_AP_dist_0.5: 0.2573  ...
  Per-class AP is the mean of the four _AP_dist_ entries.

Usage:
    python scripts/per_class_ap_plot.py
"""

import re, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

LOG_DIR    = Path(r"E:\bev_research\logs")
OUTPUT_DIR = Path(r"E:\bev_research\figures")
OUTPUT_DIR.mkdir(exist_ok=True)

BOSTON_LOG = LOG_DIR / "E2_boston_eval_retry.log"
SING_LOG   = LOG_DIR / "singapore_eval_epoch2_fixed.log"

# Canonical nuScenes class order and display labels
CLASS_ORDER = [
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
]
CLASS_LABELS = [
    "Car", "Truck", "Bus", "Trailer", "Constr.\nVeh.",
    "Pedestrian", "Motorcycle", "Bicycle", "Traffic\nCone", "Barrier",
]


def parse_per_class_ap(log_path: Path) -> dict:
    """
    Parse per-class AP from the actual MMDetection3D nuScenes log format.
    The big metrics line contains entries like:
        car_AP_dist_0.5: 0.2573  ... car_AP_dist_4.0: 0.8739
    Per-class AP = mean of the four distance-threshold values.
    """
    # Log files are UTF-16 LE (Windows Tee-Object output)
    text = log_path.read_text(encoding="utf-16", errors="replace")
    results = {}
    for cls in CLASS_ORDER:
        pat = re.compile(
            rf"{re.escape(cls)}_AP_dist_[\d.]+:\s*([\d.]+)", re.IGNORECASE
        )
        ap_vals = [float(v) for v in pat.findall(text)]
        if ap_vals:
            results[cls] = float(np.mean(ap_vals))
    return results


def main():
    boston = parse_per_class_ap(BOSTON_LOG)
    sing   = parse_per_class_ap(SING_LOG)

    if not boston:
        print("[ERROR] No per-class AP found in Boston log. Check log path and format.")
        return
    if not sing:
        print("[ERROR] No per-class AP found in Singapore log. Check log path and format.")
        return

    print("Parsed Boston classes :", sorted(boston.keys()))
    print("Parsed Singapore classes:", sorted(sing.keys()))

    # Build arrays in canonical order
    b_vals, s_vals, gaps, rel_gaps, valid_classes, valid_labels = [], [], [], [], [], []
    for cls, lbl in zip(CLASS_ORDER, CLASS_LABELS):
        if cls in boston and cls in sing:
            b = boston[cls]
            s = sing[cls]
            b_vals.append(b)
            s_vals.append(s)
            gaps.append(s - b)
            rel_gaps.append((s - b) / b * 100 if b > 0 else 0.0)
            valid_classes.append(cls)
            valid_labels.append(lbl)

    b_vals   = np.array(b_vals)
    s_vals   = np.array(s_vals)
    gaps     = np.array(gaps)
    rel_gaps = np.array(rel_gaps)
    n        = len(valid_labels)
    x        = np.arange(n)
    w        = 0.35

    # ── Figure: Grouped bar + relative degradation ───────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [2, 1]})

    ax1 = axes[0]
    bars_b = ax1.bar(x - w/2, b_vals, w, label="Boston (source)",
                     color="#2166ac", alpha=0.85, zorder=3)
    bars_s = ax1.bar(x + w/2, s_vals, w, label="Singapore (unseen)",
                     color="#d6604d", alpha=0.85, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_labels, fontsize=10)
    ax1.set_ylabel("Average Precision (AP)", fontsize=11)
    ax1.set_title(
        "Per-Class AP: Boston (source domain) vs Singapore (unseen city)\n"
        "BEVFormer-Base | nuScenes val split",
        fontsize=12, fontweight="bold"
    )
    ax1.legend(fontsize=10, loc="upper right")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax1.set_ylim(0, max(b_vals.max(), s_vals.max()) * 1.18)
    ax1.axhline(b_vals.mean(), color="#2166ac", linestyle="--",
                linewidth=1.2, alpha=0.7,
                label=f"Boston mean = {b_vals.mean():.3f}")
    ax1.axhline(s_vals.mean(), color="#d6604d", linestyle="--",
                linewidth=1.2, alpha=0.7,
                label=f"Singapore mean = {s_vals.mean():.3f}")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    # Add value labels on bars
    for bar in bars_b:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7, color="#2166ac")
    for bar in bars_s:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7, color="#d6604d")

    ax2 = axes[1]
    colors = ["#d6604d" if g < 0 else "#4dac26" for g in rel_gaps]
    bars2 = ax2.bar(x, rel_gaps, color=colors, alpha=0.85, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_labels, fontsize=10)
    ax2.set_ylabel("Relative AP Change (%)", fontsize=11)
    ax2.set_title("Relative Degradation per Class (Singapore vs Boston)", fontsize=11)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    for i, val in enumerate(rel_gaps):
        offset = -1.2 if val < 0 else 0.5
        va     = "top" if val < 0 else "bottom"
        ax2.text(i, val + offset, f"{val:+.1f}%",
                 ha="center", va=va, fontsize=8)

    plt.tight_layout()

    out_pdf = OUTPUT_DIR / "per_class_ap_comparison.pdf"
    out_png = OUTPUT_DIR / "per_class_ap_comparison.png"
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved -> {out_pdf}")
    print(f"Figure saved -> {out_png}")
    plt.close()

    # ── Ranked degradation table ──────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  {'Class':<22} {'Boston AP':>10} {'Sing AP':>9} "
          f"{'Abs Gap':>9} {'Rel Gap':>10}")
    print("  " + "-" * 63)
    order = np.argsort(rel_gaps)    # worst (most negative) first
    for i in order:
        cls = valid_classes[i]
        print(f"  {cls:<22} {b_vals[i]:>10.4f} {s_vals[i]:>9.4f} "
              f"{gaps[i]:>+9.4f} {rel_gaps[i]:>+9.1f}%")
    print(f"\n  {'mAP (mean)':<22} {b_vals.mean():>10.4f} "
          f"{s_vals.mean():>9.4f} {gaps.mean():>+9.4f} {rel_gaps.mean():>+9.1f}%")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_data = {
        cls: {
            "boston":       float(b),
            "singapore":    float(s),
            "abs_gap":      float(g),
            "rel_gap_pct":  float(r),
        }
        for cls, b, s, g, r in zip(valid_classes, b_vals, s_vals, gaps, rel_gaps)
    }
    json_path = LOG_DIR / "per_class_ap.json"
    with open(json_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nData saved -> {json_path}")


if __name__ == "__main__":
    main()
