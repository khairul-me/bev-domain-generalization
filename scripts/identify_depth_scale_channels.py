"""
identify_depth_scale_channels.py

Find which DAv2 ViT-S feature channels correlate most strongly with
depth scale (log-std of depth map, domain-invariant per stability check)
vs. edge structure (grad statistics, domain-variant).

These "invariant" channels are the safe ones to inject via the adapter.
The top 96 / 384 channels (25%) are saved to a JSON file.

FIX vs. guide original:
  - model.pretrained.blocks[-1] assumes DepthAnythingV2.pretrained is the ViT.
    In the actual depth_anything_v2 package the ViT encoder is at model.pretrained
    (a timm DINOv2 ViT), and .blocks is correct. BUT: the model.forward() triggers
    the full DPT pipeline (encoder + decoder), and we need the encoder output before
    the decoder. We hook model.pretrained.norm (the final LayerNorm of the ViT).
  - Corrected PKL paths to E:\datasets\nuscenes\...
  - Added sys.path to find the Depth-Anything-V2 package.
  - Added checks for image load failure and NaN features.

Usage:
    conda activate bev310
    cd E:\Auto_Image\bev_research\mmdetection3d
    python E:\bev_research\scripts\identify_depth_scale_channels.py --n_frames 100
"""

import argparse, json, pickle, sys
import numpy as np
import torch
import cv2
from pathlib import Path

SCRIPTS_DIR    = Path(__file__).resolve().parent
DAV2_REPO      = Path(r"E:\Auto_Image\bev_research\Depth-Anything-V2")
DATAROOT       = Path(r"E:\datasets\nuscenes")
CHECKPOINT_DAV2= Path(r"E:\bev_research\checkpoints\depth_anything_v2_vits.pth")
OUTPUT_DEFAULT = Path(r"E:\bev_research\data\dav2_channel_analysis.json")

sys.path.insert(0, str(DAV2_REPO))


def load_dav2_model(device: str = "cuda:0"):
    """Load DAv2 ViT-S in eval mode."""
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(
        encoder="vits", features=64,
        out_channels=[48, 96, 192, 384],
    )
    state = torch.load(str(CHECKPOINT_DAV2), map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def compute_depth_stats(depth: np.ndarray) -> dict:
    """The four statistics used in the original dav2_stability_check."""
    log_std  = float(np.log(float(depth.std()) + 1e-6))
    grad_x   = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y   = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return {
        "log_std":   log_std,
        "grad_mean": float(grad_mag.mean()),
        "grad_p90":  float(np.percentile(grad_mag, 90)),
    }


def load_infos(pkl_path: Path) -> list:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict):
        return d.get("infos", d.get("data_list", []))
    return d


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_dav2_model(device)

    # ── Hook on ViT final LayerNorm (post-encoder, pre-decoder) ──────────────
    # In the DepthAnythingV2 architecture: model.pretrained is the ViT.
    # model.pretrained.norm is the final LayerNorm applied after all blocks.
    # Output shape: (B, N_patches+1, 384) where N_patches = (H/14)*(W/14)
    feat_cache = {}
    def hook_fn(module, inp, out):
        # out: (B, N, 384)
        # Take the patch tokens only (exclude CLS), then global average pool -> (B, 384)
        feat_cache["feat"] = out[:, 1:, :].mean(dim=1).detach().cpu().float()

    if not hasattr(model.pretrained, "norm"):
        raise AttributeError(
            "model.pretrained has no 'norm' attribute. "
            "Check DAv2 package version or adjust hook target."
        )
    hook = model.pretrained.norm.register_forward_hook(hook_fn)

    # ── Load frame infos ──────────────────────────────────────────────────────
    boston_infos = load_infos(Path(args.boston_pkl))
    sing_infos   = load_infos(Path(args.singapore_pkl))
    n_each       = args.n_frames // 2
    all_infos    = boston_infos[:n_each] + sing_infos[:n_each]
    print(f"Processing {len(all_infos)} frames ({n_each} Boston + {n_each} Singapore)")

    all_features  = []
    all_log_std   = []
    all_grad_mean = []
    skipped       = 0

    # DAv2 normalization constants
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    with torch.no_grad():
        for i, info in enumerate(all_infos):
            if i % 20 == 0:
                print(f"  [{i:3d}/{len(all_infos)}]")

            cam_path = info.get("cams", {}).get("CAM_FRONT", {}).get("data_path", "")
            if not cam_path:
                skipped += 1
                continue
            img = cv2.imread(cam_path)
            if img is None:
                skipped += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w    = img_rgb.shape[:2]
            # DAv2 ViT-S was trained with 518×518 inputs
            inp_size = 518
            img_rs   = cv2.resize(img_rgb, (inp_size, inp_size))
            t        = (torch.from_numpy(img_rs).permute(2, 0, 1)
                        .float().unsqueeze(0) / 255.0).to(device)
            t        = (t - mean) / std

            depth = model(t).squeeze().cpu().numpy()   # (H, W)
            stats = compute_depth_stats(depth)

            feat_vec = feat_cache.get("feat")
            if feat_vec is None or feat_vec.shape[-1] != 384:
                skipped += 1
                continue
            feat_np = feat_vec.squeeze().numpy()   # (384,)
            if np.isnan(feat_np).any():
                skipped += 1
                continue

            all_features.append(feat_np)
            all_log_std.append(stats["log_std"])
            all_grad_mean.append(stats["grad_mean"])

    hook.remove()
    print(f"\nCollected {len(all_features)} valid frames  ({skipped} skipped)")

    if len(all_features) < 20:
        print("[ERROR] Too few valid frames for correlation analysis (need >=20).")
        return

    X  = np.stack(all_features)   # (n, 384)
    ls = np.array(all_log_std)     # (n,)
    gm = np.array(all_grad_mean)   # (n,)

    # ── Per-channel Pearson correlation ───────────────────────────────────────
    n_channels    = X.shape[1]
    corr_logstd   = np.array([np.corrcoef(X[:, c], ls)[0, 1]
                               for c in range(n_channels)])
    corr_gradmean = np.array([np.corrcoef(X[:, c], gm)[0, 1]
                               for c in range(n_channels)])

    # Replace NaN (zero-variance channels) with 0
    corr_logstd   = np.nan_to_num(corr_logstd)
    corr_gradmean = np.nan_to_num(corr_gradmean)

    # Invariant score: high depth-scale correlation, low edge correlation
    # Higher score = more domain-invariant for injection
    invariant_score = np.abs(corr_logstd) - np.abs(corr_gradmean)
    n_invariant     = n_channels // 4   # top 25% = 96 channels
    top_invariant   = np.argsort(-invariant_score)[:n_invariant].tolist()
    top_variant     = np.argsort(invariant_score)[:n_invariant].tolist()

    results = {
        "n_frames":              len(all_features),
        "n_channels":            int(n_channels),
        "n_invariant_channels":  n_invariant,
        "top_invariant_channels": top_invariant,   # use these for injection
        "top_variant_channels":   top_variant,     # avoid these
        "stats": {
            "corr_logstd_mean_abs":   float(np.abs(corr_logstd).mean()),
            "corr_gradmean_mean_abs": float(np.abs(corr_gradmean).mean()),
            "invariant_score_mean":   float(invariant_score.mean()),
            "invariant_score_std":    float(invariant_score.std()),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nChannel analysis saved -> {out_path}")
    print(f"Top 10 invariant channels: {top_invariant[:10]}")
    print(f"Corr with log_std  (mean |r|): {results['stats']['corr_logstd_mean_abs']:.4f}")
    print(f"Corr with grad_mean(mean |r|): {results['stats']['corr_gradmean_mean_abs']:.4f}")
    print(f"\nNext: modify depth_feature_adapter.py to inject only these {n_invariant} channels.")
    print("  f_dav2[:, self.invariant_channels, :, :]  (replace full 384-ch slice)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boston_pkl",    default=str(DATAROOT / "nuscenes_infos_temporal_val_boston.pkl"))
    parser.add_argument("--singapore_pkl", default=str(DATAROOT / "nuscenes_infos_temporal_val_singapore.pkl"))
    parser.add_argument("--n_frames",      type=int, default=100)
    parser.add_argument("--output",        default=str(OUTPUT_DEFAULT))
    args = parser.parse_args()
    main(args)
