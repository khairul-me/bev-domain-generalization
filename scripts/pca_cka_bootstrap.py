"""
PCA-reduced + debiased-HSIC CKA bootstrap CI computation.

Two-stage fix for the d >> n CI bias problem:
  1. PCA to 100D before CKA (eliminates the large-d Gram-matrix rank issue)
  2. Debiased HSIC estimator (Kornblith et al. 2019 supp.) replaces the biased
     centering formula. The biased estimator has E[HSIC_biased] = HSIC + O(1/n),
     so smaller bootstrap subsamples give systematically higher values (CIs above
     the point estimate). The debiased estimator has E[HSIC_debiased] = HSIC
     exactly, making standard bootstrap CIs valid.

Inputs:  E:\bev_research\data\cka_features_500_v2.npz
Outputs: E:\bev_research\logs\cka_pca_bootstrap.json
         (CKA point estimates + valid 95% bootstrap CIs)

CPU-only — no GPU needed.
"""

import numpy as np
import json
from sklearn.decomposition import PCA
from pathlib import Path


# ── Debiased HSIC / CKA implementation ──────────────────────────────────────
# Reference: Kornblith et al. (2019), NeurIPS, Appendix A.

def _debiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
    """
    Unbiased estimator of HSIC (Song et al. 2012, eq. used in Kornblith 2019).
    E[debiased_HSIC] = HSIC(k), unlike the biased centering estimator where
    E[biased_HSIC] = HSIC + O(1/n).
    """
    n = K.shape[0]
    K_ = K.copy(); np.fill_diagonal(K_, 0.0)
    L_ = L.copy(); np.fill_diagonal(L_, 0.0)

    ksum = K_.sum(axis=1)  # row sums, shape (n,)
    lsum = L_.sum(axis=1)

    # Three terms from the unbiased estimator formula
    term1 = float((K_ * L_).sum())
    term2 = float(ksum.sum() * lsum.sum()) / ((n - 1) * (n - 2))
    term3 = float(2.0 * ksum @ lsum) / (n - 2)

    return (term1 + term2 - term3) / (n * (n - 3))


def linear_cka_debiased(X: np.ndarray, Y: np.ndarray) -> float:
    """Debiased linear CKA between two (N, D) feature matrices."""
    K = X @ X.T
    L = Y @ Y.T
    hsic_xy = _debiased_hsic(K, L)
    hsic_xx = _debiased_hsic(K, K)
    hsic_yy = _debiased_hsic(L, L)
    denom = np.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
    if denom == 0.0:
        return 0.0
    return float(hsic_xy / denom)


# Biased version retained for comparison only
def centering(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka_biased(X: np.ndarray, Y: np.ndarray) -> float:
    """Biased linear CKA (original formulation — for comparison only)."""
    K = X @ X.T
    L = Y @ Y.T
    Kc = centering(K)
    Lc = centering(L)
    hsic_xy = np.sum(Kc * Lc)
    hsic_xx = np.sum(Kc * Kc)
    hsic_yy = np.sum(Lc * Lc)
    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


# ── Bootstrap CI (subsampling without replacement, valid with debiased CKA) ──

def bootstrap_cka_ci(X: np.ndarray, Y: np.ndarray,
                     n_boot: int = 2000, frac: float = 0.80,
                     seed: int = 42) -> tuple[float, list[float], np.ndarray]:
    """
    Debiased-CKA point estimate + 95% CI via subsampling without replacement.

    With the debiased HSIC estimator, E[CKA(subsample)] ≈ E[CKA(full)],
    so the bootstrap distribution correctly centres on the point estimate.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(int(n * frac), 4)  # need at least 4 for the n*(n-3) denominator

    point = linear_cka_debiased(X, Y)

    boot_vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=k, replace=False)
        boot_vals.append(linear_cka_debiased(X[idx], Y[idx]))

    boot_vals = np.array(boot_vals)
    ci_lo = float(np.percentile(boot_vals, 2.5))
    ci_hi = float(np.percentile(boot_vals, 97.5))
    return point, [ci_lo, ci_hi], boot_vals


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cache_path = Path(r"E:\bev_research\data\cka_features_500_v2.npz")
    out_path   = Path(r"E:\bev_research\logs\cka_pca_bootstrap.json")

    print("Loading cached features ...", flush=True)
    cache = np.load(cache_path)
    boston_img = cache["boston_img"].astype(np.float64)   # (500, 16384)
    sing_img   = cache["sing_img"].astype(np.float64)
    boston_bev = cache["boston_bev"].astype(np.float64)
    sing_bev   = cache["sing_bev"].astype(np.float64)
    print(f"  boston_img: {boston_img.shape}  sing_img: {sing_img.shape}", flush=True)

    # ── PCA reduction to 100D (d << N=500) ──────────────────────────────────
    N_COMPONENTS = 100
    print(f"\nFitting PCA to {N_COMPONENTS}D on img_feat ...", flush=True)
    pca_img = PCA(n_components=N_COMPONENTS, random_state=42)
    joint_img = np.vstack([boston_img, sing_img])   # (1000, 16384)
    pca_img.fit(joint_img)
    var_img = float(np.sum(pca_img.explained_variance_ratio_) * 100)
    boston_img_r = pca_img.transform(boston_img)    # (500, 100)
    sing_img_r   = pca_img.transform(sing_img)
    print(f"  Variance explained (img_feat): {var_img:.1f}%", flush=True)

    print(f"\nFitting PCA to {N_COMPONENTS}D on bev_embed ...", flush=True)
    pca_bev = PCA(n_components=N_COMPONENTS, random_state=42)
    joint_bev = np.vstack([boston_bev, sing_bev])
    pca_bev.fit(joint_bev)
    var_bev = float(np.sum(pca_bev.explained_variance_ratio_) * 100)
    boston_bev_r = pca_bev.transform(boston_bev)
    sing_bev_r   = pca_bev.transform(sing_bev)
    print(f"  Variance explained (bev_embed): {var_bev:.1f}%", flush=True)

    # ── CKA point estimates on PCA features ──────────────────────────────────
    print("\nComputing CKA point estimates on PCA features ...", flush=True)
    cka_img_biased   = linear_cka_biased(boston_img_r, sing_img_r)
    cka_bev_biased   = linear_cka_biased(boston_bev_r, sing_bev_r)
    cka_img_debiased = linear_cka_debiased(boston_img_r, sing_img_r)
    cka_bev_debiased = linear_cka_debiased(boston_bev_r, sing_bev_r)
    print(f"  img_feat  CKA biased   (PCA-100): {cka_img_biased:.6f}", flush=True)
    print(f"  img_feat  CKA debiased (PCA-100): {cka_img_debiased:.6f}", flush=True)
    print(f"  bev_embed CKA biased   (PCA-100): {cka_bev_biased:.6f}", flush=True)
    print(f"  bev_embed CKA debiased (PCA-100): {cka_bev_debiased:.6f}", flush=True)

    # ── Also report raw (full-dim biased) point estimates for reference ───────
    print("\nComputing CKA point estimates on original 16384D features ...", flush=True)
    cka_img_raw = linear_cka_biased(boston_img.astype(np.float32), sing_img.astype(np.float32))
    cka_bev_raw = linear_cka_biased(boston_bev.astype(np.float32), sing_bev.astype(np.float32))
    print(f"  img_feat  CKA (raw 16384D biased): {cka_img_raw:.6f}", flush=True)
    print(f"  bev_embed CKA (raw 16384D biased): {cka_bev_raw:.6f}", flush=True)

    cka_img_full  = cka_img_debiased   # use debiased for all downstream work
    cka_bev_full  = cka_bev_debiased

    # ── Bootstrap CIs on PCA-reduced features ────────────────────────────────
    N_BOOT = 2000
    print(f"\nBootstrap CIs (N={N_BOOT} iterations, 80% subsampling, no replace) ...",
          flush=True)

    print("  Running img_feat bootstrap ...", flush=True)
    img_pt, img_ci, img_boot = bootstrap_cka_ci(boston_img_r, sing_img_r, N_BOOT)
    print(f"  img_feat  CKA = {img_pt:.6f}  95% CI = [{img_ci[0]:.6f}, {img_ci[1]:.6f}]",
          flush=True)
    ci_valid_img = img_ci[0] <= img_pt <= img_ci[1]
    print(f"  CI contains point estimate: {ci_valid_img}", flush=True)

    print("  Running bev_embed bootstrap ...", flush=True)
    bev_pt, bev_ci, bev_boot = bootstrap_cka_ci(boston_bev_r, sing_bev_r, N_BOOT)
    print(f"  bev_embed CKA = {bev_pt:.6f}  95% CI = [{bev_ci[0]:.6f}, {bev_ci[1]:.6f}]",
          flush=True)
    ci_valid_bev = bev_ci[0] <= bev_pt <= bev_ci[1]
    print(f"  CI contains point estimate: {ci_valid_bev}", flush=True)

    # ── Compute gap normalization % using PCA-reduced CKA ────────────────────
    # pct = change_in_dissimilarity / initial_dissimilarity
    # dissimilarity proxy: 1 - CKA
    gap_norm_pct = (cka_bev_full - cka_img_full) / (1.0 - cka_img_full) * 100
    print(f"\nCKA gap normalization (PCA-100): {gap_norm_pct:.2f}%", flush=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    results = {
        "method": "PCA-100 + debiased HSIC estimator (Kornblith 2019 supp.)",
        "n_pairs": 500,
        "pca_components": N_COMPONENTS,
        "pca_variance_explained": {
            "img_feat": round(var_img, 2),
            "bev_embed": round(var_bev, 2)
        },
        "img_feat": {
            "linear_cka_raw_16384d_biased":   round(cka_img_raw, 6),
            "linear_cka_pca100d_biased":      round(cka_img_biased, 6),
            "linear_cka_pca100d_debiased":    round(cka_img_debiased, 6),
            "linear_cka_ci_95":               [round(img_ci[0], 6), round(img_ci[1], 6)],
            "ci_valid":                        bool(ci_valid_img)
        },
        "bev_embed": {
            "linear_cka_raw_16384d_biased":   round(cka_bev_raw, 6),
            "linear_cka_pca100d_biased":      round(cka_bev_biased, 6),
            "linear_cka_pca100d_debiased":    round(cka_bev_debiased, 6),
            "linear_cka_ci_95":               [round(bev_ci[0], 6), round(bev_ci[1], 6)],
            "ci_valid":                        bool(ci_valid_bev)
        },
        "gap_normalization_pct_by_bev_encoder_cka_pca": round(gap_norm_pct, 2)
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)

    if not ci_valid_img or not ci_valid_bev:
        print("\n[WARNING] One or more CIs do not contain the point estimate.",
              "Check feature scaling or increase N_COMPONENTS.", flush=True)
    else:
        print("\n[OK] Both CIs are valid (point estimate is inside CI bounds).", flush=True)


if __name__ == "__main__":
    main()
