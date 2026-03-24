"""
cka.py
Centered Kernel Alignment (CKA) implementation.
Reference: Kornblith et al., ICML 2019 -- "Similarity of Neural Network Representations Revisited."

This module is imported by representation_analysis_v2.py.
Run standalone for sanity checks.
"""

import numpy as np


def centering_matrix(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix: K_c = HKH  where  H = I - (1/n) 11^T."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two feature matrices.

    Args:
        X: (n_samples, d1) -- features from population A (e.g., Boston).
        Y: (n_samples, d2) -- features from population B (e.g., Singapore).
           X and Y must be sample-for-sample matched (same n_samples).

    Returns:
        CKA scalar in [0, 1].  1.0 = identical,  0.0 = unrelated.

    Properties:
        - Invariant to orthogonal transformations and isotropic scaling.
        - Not invariant to anisotropic rescaling (use rbf_cka if needed).
    """
    K  = X @ X.T    # (n, n)
    L  = Y @ Y.T    # (n, n)
    Kc = centering_matrix(K)
    Lc = centering_matrix(L)

    hsic_xy = np.sum(Kc * Lc)
    hsic_xx = np.sum(Kc * Kc)
    hsic_yy = np.sum(Lc * Lc)

    if hsic_xx == 0 or hsic_yy == 0:
        return float("nan")
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """
    RBF kernel CKA.  More robust for high-dimensional or nonlinearly structured features.
    sigma defaults to the median pairwise distance heuristic on X.
    """
    def rbf_kernel(Z: np.ndarray, sig: float) -> np.ndarray:
        sq_dists = (
            np.sum(Z ** 2, axis=1, keepdims=True)
            + np.sum(Z ** 2, axis=1)
            - 2 * Z @ Z.T
        )
        sq_dists = np.maximum(sq_dists, 0.0)    # clip numerical negatives
        return np.exp(-sq_dists / (2 * sig ** 2))

    if sigma is None:
        sq = (
            np.sum(X ** 2, axis=1, keepdims=True)
            + np.sum(X ** 2, axis=1)
            - 2 * X @ X.T
        )
        sq = np.maximum(sq, 0.0)
        positive = sq[sq > 0]
        if len(positive) == 0:
            return float("nan")
        sigma = float(np.sqrt(np.median(positive)))

    K  = rbf_kernel(X, sigma)
    L  = rbf_kernel(Y, sigma)
    Kc = centering_matrix(K)
    Lc = centering_matrix(L)

    hsic_xy = np.sum(Kc * Lc)
    hsic_xx = np.sum(Kc * Kc)
    hsic_yy = np.sum(Lc * Lc)
    if hsic_xx == 0 or hsic_yy == 0:
        return float("nan")
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def cosine_similarity_mean(X: np.ndarray, Y: np.ndarray) -> tuple:
    """
    Mean cosine similarity between matched row pairs.
    Kept for backward comparison with the original N=20 analysis.

    Returns:
        (mean, std) of per-pair cosine similarities.
    """
    X_norm  = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y_norm  = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.sum(X_norm * Y_norm, axis=1)
    return float(cos_sim.mean()), float(cos_sim.std())


# ── Sanity checks ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # For a clean "near zero" result, need N >> D substantially.
    # Using N=500, d=16 ensures the sample Gram matrices are nearly diagonal.
    N, D = 500, 16

    X = rng.standard_normal((N, D)).astype(np.float32)

    # 1. CKA(X, X) == 1.0
    val = linear_cka(X, X)
    print(f"linear_cka(X, X)              = {val:.6f}  (expect 1.0)")
    assert abs(val - 1.0) < 1e-4, f"FAILED: {val}"

    # 2. CKA(X, random Y) ~0 -- only valid when N >> D
    Y = rng.standard_normal((N, D)).astype(np.float32)
    val = linear_cka(X, Y)
    print(f"linear_cka(X, random_Y)       = {val:.6f}  (expect ~0; valid when N>>D)")
    assert val < 0.1, f"FAILED: {val}"

    # 3. CKA(X, X@Q) == 1.0 for orthogonal Q -- tests the actual invariance guarantee
    # NOTE from guide: guide used a random A; a random matrix is NOT orthogonal,
    # so CKA(X, X@A) != 1 in general. This test uses the correct orthogonal matrix.
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)).astype(np.float32))  # orthogonal
    XQ  = X @ Q
    val = linear_cka(X, XQ)
    print(f"linear_cka(X, X@Q_orthogonal) = {val:.6f}  (expect 1.0)")
    assert abs(val - 1.0) < 1e-3, f"FAILED: {val}"

    # 4. RBF CKA sanity
    val_rbf = rbf_cka(X, X)
    print(f"rbf_cka(X, X)                 = {val_rbf:.6f}  (expect 1.0)")
    assert abs(val_rbf - 1.0) < 1e-3, f"FAILED: {val_rbf}"

    # 5. Cosine similarity of X with itself == 1.0
    cos_m, cos_s = cosine_similarity_mean(X, X)
    print(f"cosine_similarity_mean(X, X)  = {cos_m:.6f} +/- {cos_s:.6f}  (expect 1.0 +/- 0.0)")
    assert abs(cos_m - 1.0) < 1e-5, f"FAILED: {cos_m}"

    print("\nAll CKA sanity checks PASSED.")
