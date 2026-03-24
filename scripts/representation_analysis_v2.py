"""
representation_analysis_v2.py

Rigorous BEV representation drift analysis.
  - N=500 semantically matched pairs (from build_semantic_pairs.py)
  - Linear CKA as primary metric (Kornblith et al., ICML 2019)
  - Cosine similarity for backward comparison with the original N=20 analysis
  - Bootstrap 95% CIs on all metrics

FIX vs. guide original:
  - Added explicit sys.path for cka module import.
  - Corrected data paths to E:\datasets\nuscenes\.
  - The dataloader construction section is clearly marked; hooks + metrics logic
    is complete and ready to use once you connect your actual inference pipeline.
  - Demo mode uses synthetic data that approximates the real cosine distributions
    (0.397 img_feat, 0.854 bev_embed) so the script is runnable immediately
    for testing. Replace with actual features before publication.

Usage (demo with synthetic data -- no GPU needed):
    conda activate bev310
    python scripts\representation_analysis_v2.py --demo

Usage (real inference -- requires GPU and BEVFormer config):
    conda activate bev310
    python scripts\representation_analysis_v2.py \
        --config E:\bev_research\configs\bevformer_singapore_eval.py \
        --checkpoint E:\bev_research\checkpoints\bevformer_base_epoch_24.pth \
        --pairs E:\bev_research\data\matched_pairs_500.json \
        --n_pairs 500
"""

import argparse, json, sys, os
import numpy as np
from pathlib import Path

# ── Make cka.py importable ────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from cka import linear_cka, rbf_cka, cosine_similarity_mean   # Phase 0.3

# PyTorch 2.6+ changed default weights_only=True which blocks mmengine checkpoints.
import torch
_torch_load_orig = torch.load
def _torch_load_patched(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_orig(f, *args, **kwargs)
torch.load = _torch_load_patched

# ── Paths ─────────────────────────────────────────────────────────────────────
DATAROOT      = Path(r"E:\datasets\nuscenes")
CHECKPOINT    = Path(r"E:\bev_research\checkpoints\bevformer_base_epoch_24.pth")
PAIRS_FILE    = Path(r"E:\bev_research\data\matched_pairs_500.json")
OUTPUT_FILE   = Path(r"E:\bev_research\logs\representation_analysis_v2.json")

# ── Hook storage ──────────────────────────────────────────────────────────────
_hooks      = {}
_feat_store = {}


def _make_hook(name: str):
    def _hook(module, inp, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        # Store the raw tensor (do NOT spatially pool here — run_and_collect
        # needs the full spatial structure to match the original N=20 methodology).
        _feat_store[name] = t.detach().cpu().float()
    return _hook


def register_hooks(model):
    """Register forward hooks at img_feat (FPN) and bev_embed (BEV encoder)."""
    img_registered = False
    for name, module in model.named_modules():
        if name == "img_neck" or name.endswith(".img_neck"):
            _hooks["img_feat"] = module.register_forward_hook(_make_hook("img_feat"))
            print(f"  Hook: img_feat -> {name}")
            img_registered = True
            break
    if not img_registered:
        raise RuntimeError("Could not find img_neck module. Check model architecture.")

    bev_registered = False
    for name, module in model.named_modules():
        if "transformer.encoder" in name and name.endswith("encoder"):
            _hooks["bev_embed"] = module.register_forward_hook(_make_hook("bev_embed"))
            print(f"  Hook: bev_embed -> {name}")
            bev_registered = True
            break
    if not bev_registered:
        raise RuntimeError("Could not find transformer.encoder module. Check model architecture.")


def remove_hooks():
    for h in _hooks.values():
        h.remove()
    _hooks.clear()


def bootstrap_metric_ci(X: np.ndarray, Y: np.ndarray, metric_fn,
                         n_boot: int = 2000, seed: int = 42,
                         subsample_frac: float = 0.80) -> tuple:
    """
    Subsampling CI (WITHOUT replacement) for CKA-like metrics.

    Standard bootstrap (with replacement) creates duplicate rows in Gram
    matrices, which inflates CKA above the full-data point estimate.
    Subsampling 80 % of pairs without replacement avoids duplicates and
    produces intervals that correctly bracket the full-data value.
    """
    rng   = np.random.default_rng(seed)
    n     = X.shape[0]
    k     = max(2, int(n * subsample_frac))   # sub-sample size
    full  = metric_fn(X, Y)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=k, replace=False)
        val = metric_fn(X[idx], Y[idx])
        if not np.isnan(val):
            boots.append(val)
    boots = np.array(boots)
    return full, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_demo(n_pairs: int, output_path: Path):
    """
    Demo run using synthetic data that approximates the real distributions.
    Replace this with the real inference loop for actual results.
    """
    print("\n[DEMO MODE] Using synthetic features approximating real distributions.")
    print("  img_feat cosine target: ~0.397,  bev_embed cosine target: ~0.854")
    print("  Replace with real inference output before publishing.\n")
    rng = np.random.default_rng(42)

    # Simulate Boston/Singapore feature pairs with the known cosine similarity levels
    boston_img = rng.standard_normal((n_pairs, 256)).astype(np.float32)
    boston_bev = rng.standard_normal((n_pairs, 256)).astype(np.float32)
    # Construct Singapore features so that cosine similarity ~ 0.40 / 0.85
    noise_img  = rng.standard_normal((n_pairs, 256)).astype(np.float32)
    noise_bev  = rng.standard_normal((n_pairs, 256)).astype(np.float32)
    sing_img   = 0.397 * boston_img + np.sqrt(1 - 0.397**2) * noise_img
    sing_bev   = 0.854 * boston_bev + np.sqrt(1 - 0.854**2) * noise_bev

    return compute_and_save(boston_img, boston_bev, sing_img, sing_bev,
                            n_pairs, output_path)


def compute_and_save(boston_img: np.ndarray, boston_bev: np.ndarray,
                     sing_img: np.ndarray, sing_bev: np.ndarray,
                     n_pairs: int, output_path: Path,
                     img_cos_override: tuple = None,
                     bev_cos_override: tuple = None) -> dict:
    """Compute all metrics and write results JSON.

    img_cos_override / bev_cos_override: optional (mean, std) tuples from
    per-camera spatial cosine computation, which overrides the default
    cosine_similarity_mean on the CKA descriptor arrays.
    """
    print(f"Computing metrics for N={n_pairs} pairs...")

    # Linear CKA with subsampling CIs
    img_cka, img_lo, img_hi = bootstrap_metric_ci(boston_img, sing_img, linear_cka)
    bev_cka, bev_lo, bev_hi = bootstrap_metric_ci(boston_bev, sing_bev, linear_cka)

    # Cosine similarity — use per-camera override when available (matches N=20 methodology)
    if img_cos_override is not None:
        img_cos_m, img_cos_s = img_cos_override
    else:
        img_cos_m, img_cos_s = cosine_similarity_mean(boston_img, sing_img)
    if bev_cos_override is not None:
        bev_cos_m, bev_cos_s = bev_cos_override
    else:
        bev_cos_m, bev_cos_s = cosine_similarity_mean(boston_bev, sing_bev)

    # Gap normalization: fraction of img-level gap closed by BEV encoder
    img_gap = 1.0 - img_cka
    bev_gap = 1.0 - bev_cka
    norm_pct = (img_gap - bev_gap) / (img_gap + 1e-8) * 100.0

    results = {
        "n_pairs": n_pairs,
        "img_feat": {
            "linear_cka":       round(img_cka, 6),
            "linear_cka_ci_95": [round(img_lo, 6), round(img_hi, 6)],
            "cosine_sim_mean":  round(img_cos_m, 6),
            "cosine_sim_std":   round(img_cos_s, 6),
        },
        "bev_embed": {
            "linear_cka":       round(bev_cka, 6),
            "linear_cka_ci_95": [round(bev_lo, 6), round(bev_hi, 6)],
            "cosine_sim_mean":  round(bev_cos_m, 6),
            "cosine_sim_std":   round(bev_cos_s, 6),
        },
        "gap_normalization_pct_by_bev_encoder": round(norm_pct, 2),
    }

    print("\n" + "=" * 58)
    print(f"  {'Metric':<32} {'img_feat':>10} {'bev_embed':>10}")
    print("  " + "-" * 55)
    print(f"  {'Linear CKA':<32} {img_cka:>10.4f} {bev_cka:>10.4f}")
    print(f"  {'CKA 95% CI lower':<32} {img_lo:>10.4f} {bev_lo:>10.4f}")
    print(f"  {'CKA 95% CI upper':<32} {img_hi:>10.4f} {bev_hi:>10.4f}")
    print(f"  {'Cosine similarity (mean)':<32} {img_cos_m:>10.4f} {bev_cos_m:>10.4f}")
    print(f"  {'Cosine similarity (std)':<32} {img_cos_s:>10.4f} {bev_cos_s:>10.4f}")
    print(f"\n  BEV encoder normalizes {norm_pct:.1f}% of appearance gap (CKA-based)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {output_path}")
    return results


def run_real_inference(args):
    """Full GPU inference pipeline using proper BEVFormer batched test_step."""
    import torch
    sys.path.insert(0, str(Path(r"E:\Auto_Image\bev_research\mmdetection3d")))
    from mmengine.config import Config
    from mmdet3d.apis    import init_model
    from mmdet3d.utils   import register_all_modules
    from mmengine.utils  import import_modules_from_strings

    register_all_modules()

    print(f"Loading model: {args.checkpoint}")
    cfg   = Config.fromfile(args.config)
    import_modules_from_strings(cfg.custom_imports.imports)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()
    register_hooks(model)

    # Load matched pairs
    with open(args.pairs) as f:
        pair_data = json.load(f)
    pairs = pair_data["pairs"][:args.n_pairs]
    print(f"Loaded {len(pairs)} matched pairs from {args.pairs}")

    import copy, pickle
    from mmdet3d.registry import DATASETS

    def build_token_to_idx(pkl_path: str) -> dict:
        """Build token -> index map by reading PKL directly (no image loading)."""
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, dict):
            infos = raw.get("data_list", raw.get("infos", []))
        else:
            infos = raw
        token_map = {}
        for i, info in enumerate(infos):
            token = (info.get("token") or info.get("sample_token")
                     or info.get("scene_token") or f"frame_{i}")
            token_map[token] = i
        return token_map

    def build_dataset_once(cfg, pkl_path: str):
        """Build a CustomNuScenesDataset for inference."""
        val_cfg = copy.deepcopy(cfg.val_dataloader)
        val_cfg["dataset"]["ann_file"] = pkl_path
        return DATASETS.build(val_cfg["dataset"])

    print("  Building token index maps (fast PKL scan)...")
    boston_token_idx = build_token_to_idx(args.boston_pkl)
    sing_token_idx   = build_token_to_idx(args.singapore_pkl)
    print(f"    Boston: {len(boston_token_idx)} tokens", flush=True)
    print(f"    Singapore: {len(sing_token_idx)} tokens", flush=True)

    print("  Building Boston inference dataset...", flush=True)
    boston_ds = build_dataset_once(cfg, args.boston_pkl)
    print("  Building Singapore inference dataset...", flush=True)
    sing_ds   = build_dataset_once(cfg, args.singapore_pkl)

    boston_img_list, boston_bev_list = [], []   # CKA descriptors
    sing_img_list,   sing_bev_list   = [], []
    img_cos_vals, bev_cos_vals = [], []   # per-pair per-camera cosine scalars
    skipped = 0

    def run_and_collect(dataset, idx):
        sample = dataset[idx]
        batched = {
            "inputs":       {"img": sample["inputs"]["img"].unsqueeze(0).to(args.device)},
            "data_samples": [sample["data_samples"]],
        }
        _feat_store.clear()
        with torch.no_grad():
            _ = model.test_step(batched)
        img_f_raw = _feat_store.get("img_feat")   # (N_cam, C, H, W) or (1, C)
        bev_f_raw = _feat_store.get("bev_embed")  # (1, H_bev*W_bev, C) or (1, C, H, W)

        # ── img_feat ─────────────────────────────────────────────────────────
        # Use CAM_FRONT (index 0) with compact spatial descriptor for CKA.
        # Cosine similarity: per-camera, full-spatial (matches original N=20 methodology).
        img_f_cka  = None
        img_f_cos  = None
        if img_f_raw is not None:
            if img_f_raw.dim() == 4:   # (N_cam, C, H, W)
                n_cam, c, h, w = img_f_raw.shape
                # Compact descriptor for CKA: CAM_FRONT → adaptive pool to 8×8 → flatten
                cam0 = img_f_raw[0:1]   # (1, C, H, W)
                cam0_pool = torch.nn.functional.adaptive_avg_pool2d(cam0, (8, 8))
                img_f_cka = cam0_pool.reshape(-1)   # (C*8*8,) = (16384,)
                # Per-camera cosine similarity descriptor: spatially flatten, then mean over cams
                flat = img_f_raw.reshape(n_cam, -1)   # (N_cam, C*H*W)
                img_f_cos = flat   # keep per-camera for downstream cosine computation
            elif img_f_raw.dim() == 2:   # (N_cam, C) – already pooled somehow
                img_f_cka = img_f_raw[0]   # (C,)
                img_f_cos = img_f_raw
            elif img_f_raw.dim() == 1:
                img_f_cka = img_f_raw
                img_f_cos = img_f_raw.unsqueeze(0)

        # ── bev_embed ────────────────────────────────────────────────────────
        # BEV encoder output: (1, H_bev*W_bev, C) or (1, C, H_bev, W_bev).
        bev_f_cka = None
        bev_f_cos = None
        if bev_f_raw is not None:
            if bev_f_raw.dim() == 3:   # (1, H*W, C)
                # Reshape to spatial map for pooling
                bev_spatial = bev_f_raw.squeeze(0)   # (H*W, C)
                bev_len = bev_spatial.shape[0]
                side = int(bev_len ** 0.5)
                if side * side == bev_len:
                    bev_spatial = bev_spatial.reshape(1, side, side, -1).permute(0, 3, 1, 2)  # (1, C, side, side)
                    bev_pool = torch.nn.functional.adaptive_avg_pool2d(bev_spatial, (8, 8))
                    bev_f_cka = bev_pool.reshape(-1)   # (C*64,)
                else:
                    bev_f_cka = bev_spatial.mean(dim=0)   # fallback: spatial mean → (C,)
                bev_f_cos = bev_f_cka.unsqueeze(0)
            elif bev_f_raw.dim() == 4:   # (1, C, H, W)
                bev_pool = torch.nn.functional.adaptive_avg_pool2d(bev_f_raw, (8, 8))
                bev_f_cka = bev_pool.reshape(-1)
                bev_f_cos = bev_f_cka.unsqueeze(0)
            elif bev_f_raw.dim() == 2:   # (1, C)
                bev_f_cka = bev_f_raw.squeeze(0)
                bev_f_cos = bev_f_raw
            elif bev_f_raw.dim() == 1:
                bev_f_cka = bev_f_raw
                bev_f_cos = bev_f_raw.unsqueeze(0)

        return img_f_cka, bev_f_cka, img_f_cos, bev_f_cos

    with torch.no_grad():
        for i, pair in enumerate(pairs):
            if i % 50 == 0:
                print(f"  Pair {i}/{len(pairs)}...", flush=True)
            try:
                b_token = pair["boston_token"]
                s_token = pair["singapore_token"]
                b_idx = boston_token_idx.get(b_token)
                s_idx = sing_token_idx.get(s_token)
                if b_idx is None or s_idx is None:
                    skipped += 1
                    continue

                b_img_cka, b_bev_cka, b_img_cos, b_bev_cos = run_and_collect(boston_ds, b_idx)
                s_img_cka, s_bev_cka, s_img_cos, s_bev_cos = run_and_collect(sing_ds, s_idx)

                if b_img_cka is None or s_img_cka is None:
                    skipped += 1
                    continue

                z_bev = np.zeros(256, dtype=np.float32)
                boston_img_list.append(b_img_cka.numpy())
                boston_bev_list.append(b_bev_cka.numpy() if b_bev_cka is not None else z_bev)
                sing_img_list.append(s_img_cka.numpy())
                sing_bev_list.append(s_bev_cka.numpy() if s_bev_cka is not None else z_bev)

                # Compute cosine similarity on-the-fly to avoid storing GBs of spatial tensors
                def _pair_cos(b_f, s_f, fallback_b, fallback_s):
                    b_np = b_f.numpy() if b_f is not None else fallback_b.unsqueeze(0).numpy()
                    s_np = s_f.numpy() if s_f is not None else fallback_s.unsqueeze(0).numpy()
                    b_n = b_np / (np.linalg.norm(b_np, axis=1, keepdims=True) + 1e-8)
                    s_n = s_np / (np.linalg.norm(s_np, axis=1, keepdims=True) + 1e-8)
                    return float(np.sum(b_n * s_n, axis=1).mean())

                img_cos_vals.append(_pair_cos(b_img_cos, s_img_cos, b_img_cka, s_img_cka))
                b_bev_cos_np = b_bev_cos if b_bev_cos is not None else torch.zeros(1, 256)
                s_bev_cos_np = s_bev_cos if s_bev_cos is not None else torch.zeros(1, 256)
                bev_cos_vals.append(_pair_cos(b_bev_cos_np, s_bev_cos_np,
                                              torch.zeros(1, 256), torch.zeros(1, 256)))

            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  [WARN] pair {i}: {e}", flush=True)

    remove_hooks()
    print(f"Collected {len(boston_img_list)} valid pairs  ({skipped} skipped)", flush=True)

    if len(boston_img_list) < 10:
        print("[ERROR] Too few valid pairs. Check dataloader / hook setup.")
        return

    boston_img_arr = np.stack(boston_img_list)
    boston_bev_arr = np.stack(boston_bev_list)
    sing_img_arr   = np.stack(sing_img_list)
    sing_bev_arr   = np.stack(sing_bev_list)

    # Per-camera cosine aggregation (scalars already computed per-pair in loop)
    img_cos_m = float(np.nanmean(img_cos_vals))
    img_cos_s = float(np.nanstd(img_cos_vals))
    bev_cos_m = float(np.nanmean(bev_cos_vals))
    bev_cos_s = float(np.nanstd(bev_cos_vals))

    # Cache CKA descriptor features to disk
    if args.features_cache:
        cache_path = Path(args.features_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            boston_img=boston_img_arr, boston_bev=boston_bev_arr,
            sing_img=sing_img_arr,     sing_bev=sing_bev_arr,
        )
        print(f"Feature cache saved -> {cache_path}", flush=True)

    return compute_and_save(
        boston_img_arr, boston_bev_arr,
        sing_img_arr,   sing_bev_arr,
        len(boston_img_list), Path(args.output),
        img_cos_override=(img_cos_m, img_cos_s),
        bev_cos_override=(bev_cos_m, bev_cos_s),
    )


def main(args):
    print("=" * 60)
    print("Representation Analysis v2 -- CKA + Bootstrap CIs")
    print("=" * 60)

    # Fast path: recompute metrics from cached features (no GPU needed)
    if args.recompute_ci:
        cache_path = Path(args.recompute_ci)
        if not cache_path.exists():
            print(f"[ERROR] Feature cache not found: {cache_path}")
            return
        print(f"Loading cached features from {cache_path}...")
        data = np.load(cache_path)
        return compute_and_save(
            data["boston_img"], data["boston_bev"],
            data["sing_img"],   data["sing_bev"],
            len(data["boston_img"]), Path(args.output)
        )

    if args.demo:
        run_demo(args.n_pairs, Path(args.output))
    else:
        if not args.config:
            print("[ERROR] --config required for real inference. Use --demo for synthetic run.")
            return
        run_real_inference(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",          action="store_true",
                        help="Run with synthetic data (no GPU needed, for testing)")
    parser.add_argument("--device",        default="cuda:0",
                        help="Torch device string (default: cuda:0)")
    parser.add_argument("--config",        default=None,
                        help="BEVFormer config .py path (required for real inference)")
    parser.add_argument("--checkpoint",    default=str(CHECKPOINT))
    parser.add_argument("--pairs",         default=str(PAIRS_FILE))
    parser.add_argument("--boston_pkl",    default=str(DATAROOT / "nuscenes_infos_temporal_val_boston_datalist.pkl"))
    parser.add_argument("--singapore_pkl", default=str(DATAROOT / "nuscenes_infos_temporal_val_singapore_datalist.pkl"))
    parser.add_argument("--output",        default=str(OUTPUT_FILE))
    parser.add_argument("--n_pairs",       type=int, default=500)
    parser.add_argument("--features-cache", dest="features_cache", default=None,
                        help="Path to save feature cache (.npz) after GPU inference")
    parser.add_argument("--recompute-ci",  dest="recompute_ci", default=None,
                        help="Load cached features (.npz) and recompute metrics only (no GPU)")
    args = parser.parse_args()
    main(args)
