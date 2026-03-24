"""
Within-Boston CKA anchor baseline.

Measures linear CKA between pairs of Boston frames taken at DIFFERENT timesteps
(same city, different scene content). This establishes a reference point for
what CKA = 0.003/0.009 means in absolute terms:

  - If within-Boston CKA >> 0.009: cross-city CKA is far below same-city variability
    → cities share essentially no relational structure (strong claim)
  - If within-Boston CKA ≈ 0.009: cross-city gap is barely larger than normal
    intra-city temporal variation → more nuanced framing needed

Uses 200 Boston validation frames; extracts img_feat and bev_embed using the
same BEVFormer hooks as representation_analysis_v2.py, then builds 500 random
within-Boston pairs (sampled without replacement from distinct frames) and
computes debiased CKA with PCA-100 reduction.

Outputs:
  E:\bev_research\logs\within_boston_cka.json
  (cross-city and within-city CKA side-by-side for the paper)
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── PyTorch 2.6 checkpoint compatibility ────────────────────────────────────
_orig_load = torch.load
def _patched_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(f, *args, **kwargs)
torch.load = _patched_load

# ── Debiased HSIC / linear CKA ───────────────────────────────────────────────

def _debiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    K_ = K.copy(); np.fill_diagonal(K_, 0.0)
    L_ = L.copy(); np.fill_diagonal(L_, 0.0)
    ksum = K_.sum(axis=1)
    lsum = L_.sum(axis=1)
    term1 = float((K_ * L_).sum())
    term2 = float(ksum.sum() * lsum.sum()) / ((n - 1) * (n - 2))
    term3 = float(2.0 * ksum @ lsum) / (n - 2)
    return (term1 + term2 - term3) / (n * (n - 3))


def linear_cka_debiased(X: np.ndarray, Y: np.ndarray) -> float:
    K = X @ X.T
    L = Y @ Y.T
    hsic_xy = _debiased_hsic(K, L)
    hsic_xx = _debiased_hsic(K, K)
    hsic_yy = _debiased_hsic(L, L)
    denom = np.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
    return float(hsic_xy / denom) if denom > 0 else 0.0


def bootstrap_cka_ci(X: np.ndarray, Y: np.ndarray,
                     n_boot: int = 2000, frac: float = 0.80, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(int(n * frac), 4)
    point = linear_cka_debiased(X, Y)
    boot = [linear_cka_debiased(X[idx := rng.choice(n, k, replace=False)],
                                Y[idx]) for _ in range(n_boot)]
    boot = np.array(boot)
    return point, [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


# ── Feature extraction (hooks on BEVFormer) ──────────────────────────────────

def build_model_and_hooks(cfg_path: str, ckpt_path: str, device: str):
    import sys
    sys.path.insert(0, r"E:\Auto_Image\bev_research\mmdetection3d")
    from mmdet3d.apis import init_model
    from mmdet3d.utils import register_all_modules
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings

    register_all_modules()
    cfg = Config.fromfile(cfg_path)
    import_modules_from_strings(cfg.custom_imports.imports)
    model = init_model(cfg, ckpt_path, device=device)
    model.eval()

    captured = {}

    def make_hook(tag):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (list, tuple)) else out
            captured[tag] = t.detach().cpu()
        return hook

    model.img_neck.register_forward_hook(make_hook("img_feat_raw"))
    model.pts_bbox_head.transformer.encoder.register_forward_hook(
        make_hook("bev_embed_raw"))

    return model, captured


def extract_features(model, captured, dataloader, n_frames: int, device: str):
    img_feats, bev_embeds = [], []
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if count >= n_frames:
                break
            inputs = batch["inputs"]
            data_samples = batch["data_samples"]
            if isinstance(inputs, dict):
                imgs = inputs.get("imgs", inputs.get("img"))
            else:
                imgs = inputs
            imgs = imgs.to(device)
            if imgs.dim() == 4:
                imgs = imgs.unsqueeze(0)
            if not isinstance(data_samples, list):
                data_samples = [data_samples]
            try:
                model.test_step({"imgs": imgs, "data_samples": data_samples})
            except Exception:
                continue

            # img_feat: take FPN level 0, CAM_FRONT (index 0),
            # adaptive-avg-pool to 8x8, flatten → 16384D
            raw_fpn = captured.get("img_feat_raw")
            if raw_fpn is None:
                continue
            if isinstance(raw_fpn, (list, tuple)):
                raw_fpn = raw_fpn[0]
            # shape: (B*N_cam, C, H, W) — take first camera
            cam0 = raw_fpn[0:1]  # (1, C, H, W)
            pooled = F.adaptive_avg_pool2d(cam0, (8, 8))
            img_feats.append(pooled.flatten().numpy())

            # bev_embed: reshape to spatial, pool to 8x8, flatten
            raw_bev = captured.get("bev_embed_raw")
            if raw_bev is None:
                continue
            # shape: (1, H_bev*W_bev, C) or (1, C, H_bev, W_bev)
            if raw_bev.dim() == 3:
                # (1, seq, C) → (1, C, H, W)
                seq = raw_bev.shape[1]
                C   = raw_bev.shape[2]
                H   = W = int(seq ** 0.5)
                raw_bev = raw_bev.permute(0, 2, 1).reshape(1, C, H, W)
            pooled_bev = F.adaptive_avg_pool2d(raw_bev, (8, 8))
            bev_embeds.append(pooled_bev.flatten().numpy())

            captured.clear()
            count += 1
            if count % 20 == 0:
                print(f"  Extracted {count}/{n_frames} frames ...", flush=True)

    return np.stack(img_feats), np.stack(bev_embeds)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    CKPT   = r"E:\bev_research\checkpoints\bevformer_base_epoch_24.pth"
    BOSTON_PKL = r"C:\datasets\nuscenes\nuscenes_infos_temporal_val_boston_datalist.pkl"
    OUT_PATH   = Path(r"E:\bev_research\logs\within_boston_cka.json")
    N_FRAMES   = 200
    N_PAIRS    = 500

    print("=" * 64, flush=True)
    print("Within-Boston CKA Anchor Baseline", flush=True)
    print("=" * 64, flush=True)

    # ── Build dataset / dataloader ────────────────────────────────────────────
    sys.path.insert(0, r"E:\Auto_Image\bev_research\mmdetection3d")
    from mmdet3d.registry import DATASETS
    from mmdet3d.utils import register_all_modules
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings

    register_all_modules()
    cfg = Config.fromfile(r"E:\bev_research\configs\bevformer_rtx5060.py")
    import_modules_from_strings(cfg.custom_imports.imports)

    ds_cfg = cfg.val_dataloader.dataset.to_dict()
    ds_cfg["ann_file"] = BOSTON_PKL

    print("Building Boston dataset ...", flush=True)
    dataset = DATASETS.build(ds_cfg)

    from torch.utils.data import DataLoader
    from mmengine.dataset import default_collate
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=default_collate)

    CFG_PATH   = r"E:\bev_research\configs\bevformer_rtx5060.py"

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"Loading BEVFormer on {DEVICE} ...", flush=True)
    model, captured = build_model_and_hooks(CFG_PATH, CKPT, DEVICE)

    # ── Extract features for N_FRAMES Boston frames ───────────────────────────
    print(f"\nExtracting {N_FRAMES} Boston frames ...", flush=True)
    img_feats, bev_embeds = extract_features(
        model, captured, loader, N_FRAMES, DEVICE)
    print(f"  img_feats:  {img_feats.shape}", flush=True)
    print(f"  bev_embeds: {bev_embeds.shape}", flush=True)

    # ── Build N_PAIRS random within-Boston pairs (shuffled halves) ────────────
    rng = np.random.default_rng(SEED)
    idx_all = np.arange(N_FRAMES)
    rng.shuffle(idx_all)
    half = N_FRAMES // 2

    # Each "pair" is frame idx_all[i] vs idx_all[i + half] — guaranteed distinct
    pair_A = idx_all[:half]   # 100 frames
    pair_B = idx_all[half:]   # 100 frames
    n_pairs = min(len(pair_A), len(pair_B), N_PAIRS)

    img_A = img_feats[pair_A[:n_pairs]]   # (100, 16384)
    img_B = img_feats[pair_B[:n_pairs]]
    bev_A = bev_embeds[pair_A[:n_pairs]]
    bev_B = bev_embeds[pair_B[:n_pairs]]

    print(f"\nUsing {n_pairs} within-Boston pairs for CKA.", flush=True)

    # ── PCA + debiased CKA ────────────────────────────────────────────────────
    N_COMP = 100
    print(f"\nPCA to {N_COMP}D ...", flush=True)

    pca_img = PCA(n_components=N_COMP, random_state=SEED)
    pca_img.fit(np.vstack([img_A, img_B]))
    var_img = float(np.sum(pca_img.explained_variance_ratio_) * 100)
    print(f"  img_feat variance explained: {var_img:.1f}%", flush=True)
    img_A_r = pca_img.transform(img_A)
    img_B_r = pca_img.transform(img_B)

    pca_bev = PCA(n_components=N_COMP, random_state=SEED)
    pca_bev.fit(np.vstack([bev_A, bev_B]))
    var_bev = float(np.sum(pca_bev.explained_variance_ratio_) * 100)
    print(f"  bev_embed variance explained: {var_bev:.1f}%", flush=True)
    bev_A_r = pca_bev.transform(bev_A)
    bev_B_r = pca_bev.transform(bev_B)

    print("\nComputing within-Boston debiased CKA + bootstrap CIs ...", flush=True)
    wb_img_pt, wb_img_ci = bootstrap_cka_ci(img_A_r, img_B_r)
    wb_bev_pt, wb_bev_ci = bootstrap_cka_ci(bev_A_r, bev_B_r)

    print(f"  img_feat  within-Boston CKA = {wb_img_pt:.6f}  "
          f"95% CI = [{wb_img_ci[0]:.6f}, {wb_img_ci[1]:.6f}]", flush=True)
    print(f"  bev_embed within-Boston CKA = {wb_bev_pt:.6f}  "
          f"95% CI = [{wb_bev_ci[0]:.6f}, {wb_bev_ci[1]:.6f}]", flush=True)

    # ── Load cross-city results for comparison ────────────────────────────────
    cross_city_path = Path(r"E:\bev_research\logs\cka_pca_bootstrap.json")
    cross_city = {}
    if cross_city_path.exists():
        cross_city = json.loads(cross_city_path.read_text())

    cc_img_pt = cross_city.get("img_feat", {}).get("linear_cka_pca100d_debiased", None)
    cc_img_ci = cross_city.get("img_feat", {}).get("linear_cka_ci_95", [None, None])
    cc_bev_pt = cross_city.get("bev_embed", {}).get("linear_cka_pca100d_debiased", None)
    cc_bev_ci = cross_city.get("bev_embed", {}).get("linear_cka_ci_95", [None, None])

    print("\n── Cross-city vs. within-Boston summary ──────────────────────", flush=True)
    print(f"  img_feat  cross-city  = {cc_img_pt}  CI = {cc_img_ci}", flush=True)
    print(f"  img_feat  within-city = {wb_img_pt:.6f}  CI = {wb_img_ci}", flush=True)
    print(f"  bev_embed cross-city  = {cc_bev_pt}  CI = {cc_bev_ci}", flush=True)
    print(f"  bev_embed within-city = {wb_bev_pt:.6f}  CI = {wb_bev_ci}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "method": "debiased CKA + PCA-100, within-Boston temporal pairs",
        "n_frames": N_FRAMES,
        "n_pairs": n_pairs,
        "pca_components": N_COMP,
        "pca_variance_explained": {"img_feat": round(var_img, 2),
                                   "bev_embed": round(var_bev, 2)},
        "within_boston": {
            "img_feat":  {"cka": round(wb_img_pt, 6), "ci_95": [round(v, 6) for v in wb_img_ci]},
            "bev_embed": {"cka": round(wb_bev_pt, 6), "ci_95": [round(v, 6) for v in wb_bev_ci]},
        },
        "cross_city": {
            "img_feat":  {"cka": cc_img_pt, "ci_95": cc_img_ci},
            "bev_embed": {"cka": cc_bev_pt, "ci_95": cc_bev_ci},
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
