"""
bootstrap_ci.py
Compute 95% bootstrap confidence intervals on per-class AP and NDS
from BEVFormer evaluation log files.

FIX vs. guide original:
  The guide assumed log lines like "car AP: 0.5731". The actual MMDetection3D
  log format puts all metrics on ONE long line as key-value pairs:
    NuScenes metric/pred_instances_3d_NuScenes/car_AP_dist_0.5: 0.2573  ...
  Per-class AP is computed here as the mean of the four _AP_dist_ entries.
  The summary metrics (mAP, NDS, mATE...) appear on short standalone lines
  before the long line and are parsed separately.

Usage:
    python scripts/bootstrap_ci.py
"""

import numpy as np
import json, re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR     = Path(r"E:\bev_research\logs")
BOSTON_LOG  = LOG_DIR / "E2_boston_eval_retry.log"
SING_LOG    = LOG_DIR / "singapore_eval_epoch2_fixed.log"
N_BOOTSTRAP = 10_000
RNG_SEED    = 42

# nuScenes canonical classes -- order must match this list for all downstream code
CLASSES = [
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
]
# ──────────────────────────────────────────────────────────────────────────────


def parse_per_class_ap(log_path: Path) -> dict:
    """
    Parse per-class AP from the MMDetection3D nuScenes evaluation log.

    The log contains one long line like:
        ... car_AP_dist_0.5: 0.2573  ... car_AP_dist_1.0: 0.5651 ...

    Per-class AP = mean of the four distance-threshold APs (0.5, 1.0, 2.0, 4.0).
    Returns dict {class_name: mean_ap}.
    """
    # Log files are UTF-16 LE (Windows Tee-Object output)
    text = log_path.read_text(encoding="utf-16", errors="replace")
    results = {}
    for cls in CLASSES:
        # Match all four distance thresholds for this class
        pat = re.compile(
            rf"{re.escape(cls)}_AP_dist_[\d.]+:\s*([\d.]+)", re.IGNORECASE
        )
        matches = pat.findall(text)
        if matches:
            ap_vals = [float(v) for v in matches]
            results[cls] = float(np.mean(ap_vals))
    return results


def parse_summary_metrics(log_path: Path) -> dict:
    """
    Parse headline mAP, NDS and all sub-errors.
    These appear as short standalone lines before the big metrics line, e.g.:
        mAP: 0.4250
        NDS: 0.5239
    """
    metrics = {}
    patterns = {
        "mAP":  re.compile(r"^mAP:\s*([\d.]+)", re.MULTILINE),
        "NDS":  re.compile(r"^NDS:\s*([\d.]+)", re.MULTILINE),
        "mATE": re.compile(r"^mATE:\s*([\d.]+)", re.MULTILINE),
        "mASE": re.compile(r"^mASE:\s*([\d.]+)", re.MULTILINE),
        "mAOE": re.compile(r"^mAOE:\s*([\d.]+)", re.MULTILINE),
        "mAVE": re.compile(r"^mAVE:\s*([\d.]+)", re.MULTILINE),
        "mAAE": re.compile(r"^mAAE:\s*([\d.]+)", re.MULTILINE),
    }
    # Log files are UTF-16 LE (Windows Tee-Object output)
    text = log_path.read_text(encoding="utf-16", errors="replace")
    for name, pat in patterns.items():
        m = pat.search(text)
        if m:
            metrics[name] = float(m.group(1))
    return metrics


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000,
                 alpha: float = 0.05, rng_seed: int = 42) -> tuple:
    """Return (mean, lower_ci, upper_ci) via percentile bootstrap."""
    rng = np.random.default_rng(rng_seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return values.mean(), lower, upper


def bootstrap_gap_ci(boston_vals: np.ndarray, sing_vals: np.ndarray,
                     n_boot: int = 10_000, rng_seed: int = 42) -> tuple:
    """Bootstrap CI on the gap (Singapore - Boston) between two independent samples."""
    rng = np.random.default_rng(rng_seed)
    boot_gaps = np.array([
        rng.choice(sing_vals,   size=len(sing_vals),   replace=True).mean() -
        rng.choice(boston_vals, size=len(boston_vals), replace=True).mean()
        for _ in range(n_boot)
    ])
    gap   = sing_vals.mean() - boston_vals.mean()
    lower = np.percentile(boot_gaps, 2.5)
    upper = np.percentile(boot_gaps, 97.5)
    return gap, lower, upper


def main():
    print("=" * 60)
    print("Bootstrap CI Analysis - Boston vs Singapore Gap")
    print("=" * 60)

    boston_cls = parse_per_class_ap(BOSTON_LOG)
    sing_cls   = parse_per_class_ap(SING_LOG)
    boston_sum = parse_summary_metrics(BOSTON_LOG)
    sing_sum   = parse_summary_metrics(SING_LOG)

    if not boston_cls:
        print("\n[ERROR] Boston per-class AP could not be parsed.")
        print("  Check that BOSTON_LOG path is correct and the log contains")
        print("  lines matching 'car_AP_dist_0.5: ...'")
        return
    if not sing_cls:
        print("\n[ERROR] Singapore per-class AP could not be parsed.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    print("\n[1] Summary metrics (from standalone log lines):")
    print(f"  {'Metric':<10}  {'Boston':>8}  {'Singapore':>10}  {'Gap':>8}")
    print(f"  {'-'*42}")
    for k in ["mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE"]:
        b   = boston_sum.get(k, float("nan"))
        s   = sing_sum.get(k, float("nan"))
        gap = s - b
        print(f"  {k:<10}  {b:>8.4f}  {s:>10.4f}  {gap:>+8.4f}")

    # ── Per-class AP table ────────────────────────────────────────────────────
    print(f"\n[2] Per-class AP (mean over 4 distance thresholds):")
    print(f"  {'Class':<25} {'Boston':>8} {'Singapore':>10} {'Gap':>8}")
    print(f"  {'-'*54}")
    for cls in CLASSES:
        b = boston_cls.get(cls, float("nan"))
        s = sing_cls.get(cls, float("nan"))
        print(f"  {cls:<25} {b:>8.4f} {s:>10.4f} {s-b:>+8.4f}")

    # ── Bootstrap over per-class APs (N=10 classes) ───────────────────────────
    # NOTE: N=10 is a small sample; CIs are wide and conservative.
    # This is the correct bootstrap unit -- each class AP is one observation.
    common = [c for c in CLASSES if c in boston_cls and c in sing_cls]
    b_arr  = np.array([boston_cls[c] for c in common])
    s_arr  = np.array([sing_cls[c]   for c in common])

    b_mean, b_lo, b_hi = bootstrap_ci(b_arr, N_BOOTSTRAP, rng_seed=RNG_SEED)
    s_mean, s_lo, s_hi = bootstrap_ci(s_arr, N_BOOTSTRAP, rng_seed=RNG_SEED)
    gap, g_lo, g_hi    = bootstrap_gap_ci(b_arr, s_arr, N_BOOTSTRAP, RNG_SEED)

    print(f"\n[3] Bootstrap 95% CIs on mAP (mean over {len(common)} classes):")
    print(f"  Boston    : {b_mean:.4f}  [{b_lo:.4f}, {b_hi:.4f}]")
    print(f"  Singapore : {s_mean:.4f}  [{s_lo:.4f}, {s_hi:.4f}]")
    print(f"  Gap (S-B) : {gap:+.4f}  [{g_lo:+.4f}, {g_hi:+.4f}]")

    # ── Significance tests ────────────────────────────────────────────────────
    from scipy import stats
    t_stat, t_p = stats.ttest_rel(b_arr, s_arr)
    w_stat, w_p = stats.wilcoxon(b_arr, s_arr)
    print(f"\n[4] Significance tests on per-class AP (N={len(common)}):")
    print(f"  Paired t-test:        t={t_stat:.3f},  p={t_p:.4f}")
    print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.4f}")
    sig = "SIGNIFICANT (p < 0.05)" if t_p < 0.05 else "NOT significant at p < 0.05"
    print(f"  -> Gap is {sig}")
    print(f"  NOTE: N=10 classes gives low statistical power. Report CIs alongside p-values.")

    # ── Paper-ready string ────────────────────────────────────────────────────
    print(f"\n[5] Paper-ready gap row for Table 1:")
    print(f"  Gap (S-B): {gap*100:+.1f} mAP  [{g_lo*100:+.1f}, {g_hi*100:+.1f}],  "
          f"p={t_p:.3f} (paired t), p={w_p:.3f} (Wilcoxon)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "boston_summary":    boston_sum,
        "singapore_summary": sing_sum,
        "bootstrap": {
            "n_classes":        len(common),
            "classes":          common,
            "boston_mAP_ci":    [b_mean, b_lo, b_hi],
            "singapore_mAP_ci": [s_mean, s_lo, s_hi],
            "gap_ci":           [gap, g_lo, g_hi],
        },
        "significance": {
            "paired_t_stat": float(t_stat),
            "paired_t_p":    float(t_p),
            "wilcoxon_W":    float(w_stat),
            "wilcoxon_p":    float(w_p),
        },
        "boston_per_class":    boston_cls,
        "singapore_per_class": sing_cls,
    }
    out_path = LOG_DIR / "bootstrap_ci_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[6] Results saved -> {out_path}")


if __name__ == "__main__":
    main()
