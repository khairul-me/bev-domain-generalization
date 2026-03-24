"""
tsne_encoder_drift.py

Extract BEV encoder output features from:
  (A) Frozen baseline BEVFormer (bevformer_base_epoch_24.pth)
  (B) E4 checkpoint (partial unfreeze, epoch 4)

Run t-SNE on both and plot side-by-side to visualize encoder drift.
Coloring: by city (Boston/Singapore) and by model (Baseline/E4).

Also reports a drift ratio:
  cross_model_distance / within_baseline_cross_city_distance
  >> 1  ->  catastrophic forgetting (encoder moved more than the city gap)
  ~  1  ->  calibration decoupling only (encoder stayed in same space)

FIX vs. guide original:
  - Corrected E4 checkpoint path to actual location:
      E:\bev_research\experiments\E4_partial_unfreeze\subset2k\epoch_4.pth
  - Corrected data PKL paths to E:\datasets\nuscenes\.
  - inference_detector() only works with single image path; BEVFormer is
    multi-camera + temporal. We use a DataLoader approach consistent with
    how your existing eval runs work.
  - Added PCA->50 dims before t-SNE for numerical stability and speed.

Usage:
    conda activate bev310
    pip install scikit-learn
    cd E:\Auto_Image\bev_research\mmdetection3d
    python E:\bev_research\scripts\tsne_encoder_drift.py \
        --config E:\bev_research\configs\bevformer_singapore_eval.py \
        --n_frames 200
"""

import argparse, sys, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

MMDET3D_ROOT    = Path(r"E:\Auto_Image\bev_research\mmdetection3d")
SCRIPTS_DIR     = Path(__file__).resolve().parent
DATAROOT        = Path(r"E:\datasets\nuscenes")
CHECKPOINT_BASE = Path(r"E:\bev_research\checkpoints\bevformer_base_epoch_24.pth")
CHECKPOINT_E4   = Path(r"E:\bev_research\experiments\E4_partial_unfreeze\subset2k\epoch_4.pth")
# Use _datalist.pkl variants — these are what CustomNuScenesDataset expects
BOSTON_PKL      = DATAROOT / "nuscenes_infos_temporal_val_boston_datalist.pkl"
SINGAPORE_PKL   = DATAROOT / "nuscenes_infos_temporal_val_singapore_datalist.pkl"
OUTPUT_DEFAULT  = Path(r"E:\bev_research\figures\tsne_encoder_drift.pdf")

sys.path.insert(0, str(MMDET3D_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

# PyTorch 2.6+ changed default weights_only=True which blocks mmengine checkpoints.
import torch
_torch_load_orig = torch.load
def _torch_load_patched(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_orig(f, *args, **kwargs)
torch.load = _torch_load_patched


def load_infos(pkl_path: Path) -> list:
    import pickle
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict):
        return d.get("infos", d.get("data_list", []))
    return d


def extract_bev_features(model, cfg, pkl_path: str, n_frames: int,
                          device: str = "cuda:0") -> np.ndarray:
    """
    Extract spatially mean-pooled BEV encoder output for n_frames.
    Uses the mmdet3d-registered CustomNuScenesDataset and correct
    batched test_step format: {'inputs': {'img': (1,6,3,H,W)}, 'data_samples': [...]}.
    """
    import copy
    from mmengine.utils import import_modules_from_strings
    from mmdet3d.registry import DATASETS

    import_modules_from_strings(cfg.custom_imports.imports)

    features   = []
    hook_cache = {}

    def hook_fn(module, inp, out):
        t = out[0] if isinstance(out, (list, tuple)) else out
        if t.dim() == 4:      # (B, C, H, W)
            feat = t.mean(dim=[2, 3])
        else:                  # (B, H*W, C)
            feat = t.mean(dim=1)
        hook_cache["bev"] = feat.detach().cpu().float()

    handle = None
    for name, module in model.named_modules():
        if "transformer.encoder" in name and name.endswith("encoder"):
            handle = module.register_forward_hook(hook_fn)
            print(f"  Hook registered -> {name}")
            break
    if handle is None:
        raise RuntimeError("BEV encoder module not found.")

    val_cfg = copy.deepcopy(cfg.val_dataloader)
    val_cfg["dataset"]["ann_file"] = pkl_path
    dataset = DATASETS.build(val_cfg["dataset"])
    n_available = min(n_frames, len(dataset))
    print(f"  Dataset: {len(dataset)} frames, using first {n_available}")

    model.eval()
    skipped = 0
    with torch.no_grad():
        for i in range(n_available):
            if i % 50 == 0:
                print(f"    [{i}/{n_available}]")
            try:
                sample = dataset[i]
                batched = {
                    "inputs":      {"img": sample["inputs"]["img"].unsqueeze(0).to(device)},
                    "data_samples": [sample["data_samples"]],
                }
                _ = model.test_step(batched)
                bev = hook_cache.get("bev")
                if bev is not None and not torch.isnan(bev).any():
                    features.append(bev.squeeze().numpy())
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"    [WARN] frame {i}: {e}")

    handle.remove()
    print(f"  Extracted {len(features)} features  ({skipped} skipped)")
    return np.array(features) if features else np.zeros((0, 256))


def main(args):
    import torch
    from mmengine.config import Config
    from mmdet3d.apis import init_model
    from mmdet3d.utils import register_all_modules
    from mmengine.utils import import_modules_from_strings

    register_all_modules()
    print("=" * 60)
    print("t-SNE Encoder Drift Analysis: Baseline vs E4")
    print("=" * 60)

    cfg = Config.fromfile(args.config)
    import_modules_from_strings(cfg.custom_imports.imports)
    n   = args.n_frames // 2   # frames per city per model

    # ── Extract from BASELINE model ────────────────────────────────────────────
    print(f"\n[1/4] Baseline model: {args.checkpoint_base}")
    model_base = init_model(cfg, args.checkpoint_base, device=args.device)
    base_boston = extract_bev_features(model_base, cfg, args.boston_pkl, n, args.device)
    base_sing   = extract_bev_features(model_base, cfg, args.singapore_pkl, n, args.device)
    del model_base
    torch.cuda.empty_cache()

    # ── Extract from E4 model ──────────────────────────────────────────────────
    print(f"\n[2/4] E4 model: {args.checkpoint_e4}")
    if not Path(args.checkpoint_e4).exists():
        print(f"[ERROR] E4 checkpoint not found: {args.checkpoint_e4}")
        print("  Run the E4 partial-unfreeze training first.")
        return
    model_e4 = init_model(cfg, args.checkpoint_e4, device=args.device)
    e4_boston = extract_bev_features(model_e4, cfg, args.boston_pkl, n, args.device)
    e4_sing   = extract_bev_features(model_e4, cfg, args.singapore_pkl, n, args.device)
    del model_e4
    torch.cuda.empty_cache()

    # ── Validate shapes ────────────────────────────────────────────────────────
    for name, arr in [("base_boston", base_boston), ("base_sing", base_sing),
                      ("e4_boston", e4_boston), ("e4_sing", e4_sing)]:
        if arr.shape[0] == 0:
            print(f"[ERROR] {name} is empty. Check feature extraction.")
            return
    print(f"\n[3/4] Feature shapes: "
          f"base_boston={base_boston.shape}, base_sing={base_sing.shape}, "
          f"e4_boston={e4_boston.shape}, e4_sing={e4_sing.shape}")

    # ── Stack and run t-SNE ────────────────────────────────────────────────────
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    nb, ns = len(base_boston), len(base_sing)
    ne_b, ne_s = len(e4_boston), len(e4_sing)
    all_feats = np.vstack([base_boston, base_sing, e4_boston, e4_sing])

    city_labels  = (["Boston"]*nb    + ["Singapore"]*ns +
                    ["Boston"]*ne_b  + ["Singapore"]*ne_s)
    model_labels = (["Baseline"]*(nb+ns) + ["E4"]*(ne_b+ne_s))

    print(f"  Running PCA -> 50 dims on {len(all_feats)} feature vectors...")
    n_pca   = min(50, all_feats.shape[1])
    reduced = PCA(n_components=n_pca).fit_transform(all_feats)

    print(f"  Running t-SNE (perplexity=30, n_iter=1000)...")
    embed = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42,
                 n_jobs=1).fit_transform(reduced)

    # ── Plot ───────────────────────────────────────────────────────────────────
    from matplotlib.patches import Ellipse
    COLORS_CITY  = {"Boston": "#2166ac", "Singapore": "#d6604d"}
    COLORS_MODEL = {"Baseline": "#1a9641", "E4": "#d7191c"}
    MARKERS      = {"Baseline": "o", "E4": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, (title, labels, palette) in zip(axes, [
        ("Colored by City",  city_labels,  COLORS_CITY),
        ("Colored by Model", model_labels, COLORS_MODEL),
    ]):
        for label in sorted(set(labels)):
            idx  = [i for i, l in enumerate(labels) if l == label]
            mark = "o" if title == "Colored by City" else MARKERS.get(label, "o")
            ax.scatter(embed[idx, 0], embed[idx, 1],
                       c=palette[label], label=label,
                       s=12, alpha=0.55, marker=mark, zorder=2)

        # Draw 2σ ellipses for model clusters
        if title == "Colored by Model":
            for mdl, color in COLORS_MODEL.items():
                idx = [i for i, l in enumerate(model_labels) if l == mdl]
                pts = embed[idx]
                if len(pts) < 4:
                    continue
                cx, cy = pts.mean(axis=0)
                sx, sy = pts.std(axis=0) * 2.5
                ell = Ellipse((cx, cy), sx * 2, sy * 2, fill=False,
                              edgecolor=color, linewidth=1.8, linestyle="--", zorder=3)
                ax.add_patch(ell)

        ax.legend(fontsize=10, markerscale=1.5)
        ax.set_title(f"BEV Encoder Features -- t-SNE ({title})", fontsize=11)
        ax.set_xlabel("t-SNE dim 1", fontsize=10)
        ax.set_ylabel("t-SNE dim 2", fontsize=10)
        ax.grid(alpha=0.2)

    plt.suptitle(
        "E4 Encoder Drift: Does the encoder shift into a new representation space?\n"
        "(Drift ratio >> 1 -> catastrophic forgetting;  ~ 1 -> calibration decoupling)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"\nFigure saved -> {out}")
    plt.close()

    # ── Drift ratio ────────────────────────────────────────────────────────────
    n_base    = nb + ns
    base_feat = all_feats[:n_base]
    e4_feat   = all_feats[n_base:]

    cross_model     = float(np.linalg.norm(base_feat.mean(0) - e4_feat.mean(0)))
    within_baseline = float(np.linalg.norm(base_boston.mean(0) - base_sing.mean(0)))
    drift_ratio     = cross_model / (within_baseline + 1e-8)

    print(f"\n[4/4] Drift quantification:")
    print(f"  Cross-model distance (Baseline -> E4):     {cross_model:.4f}")
    print(f"  Within-baseline cross-city distance:       {within_baseline:.4f}")
    print(f"  Drift ratio:                               {drift_ratio:.2f}x")
    if drift_ratio > 1.5:
        print("  -> Drift >> city gap: CATASTROPHIC FORGETTING")
    elif drift_ratio > 0.8:
        print("  -> Drift ~ city gap: CALIBRATION DECOUPLING (encoder stayed in same space)")
    else:
        print("  -> Drift < city gap: minimal encoder shift")

    result = {
        "cross_model_distance":    cross_model,
        "within_baseline_distance": within_baseline,
        "drift_ratio":              drift_ratio,
        "interpretation": (
            "catastrophic_forgetting" if drift_ratio > 1.5
            else "calibration_decoupling" if drift_ratio > 0.8
            else "minimal_drift"
        ),
    }
    json_path = Path(args.output).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Drift metrics saved -> {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           required=True,
                        help="BEVFormer eval config (e.g. bevformer_singapore_eval.py)")
    parser.add_argument("--checkpoint_base",  default=str(CHECKPOINT_BASE))
    parser.add_argument("--checkpoint_e4",    default=str(CHECKPOINT_E4))
    parser.add_argument("--boston_pkl",       default=str(BOSTON_PKL))
    parser.add_argument("--singapore_pkl",    default=str(SINGAPORE_PKL))
    parser.add_argument("--n_frames",         type=int, default=200)
    parser.add_argument("--device",           default="cuda:0")
    parser.add_argument("--output",           default=str(OUTPUT_DEFAULT))
    args = parser.parse_args()
    main(args)
