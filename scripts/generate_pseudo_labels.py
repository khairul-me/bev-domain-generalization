"""
generate_pseudo_labels.py

Run frozen BEVFormer on Singapore validation frames.
Filter predictions by confidence threshold.
Save results as a nuScenes-format PKL for pseudo-label supervision (Phase 4.0).

FIX vs. guide original:
  - Uses mmdet3d.registry.DATASETS to build the dataset (no build_dataset API).
  - Uses the correct batched test_step format:
      {'inputs': {'img': (1,6,3,H,W)}, 'data_samples': [Det3DDataSample]}
  - register_all_modules() + import_modules_from_strings() to register CustomNuScenesDataset.
  - Corrected PKL path to _datalist variant (what CustomNuScenesDataset expects).
  - PyTorch 2.6+ weights_only patch applied before any mmengine/checkpoint use.

Usage:
    conda activate bev310
    cd E:\\Auto_Image\\bev_research\\mmdetection3d
    python E:\\bev_research\\scripts\\generate_pseudo_labels.py \\
        --config E:\\bev_research\\configs\\bevformer_singapore_eval.py \\
        --checkpoint E:\\bev_research\\checkpoints\\bevformer_base_epoch_24.pth \\
        --score_threshold 0.3
"""

import argparse, pickle, sys, json
import numpy as np
import torch
from pathlib import Path

MMDET3D_ROOT = Path(r"E:\Auto_Image\bev_research\mmdetection3d")
sys.path.insert(0, str(MMDET3D_ROOT))

# PyTorch 2.6+ changed default weights_only=True which blocks mmengine checkpoints.
_torch_load_orig = torch.load
def _torch_load_patched(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_orig(f, *args, **kwargs)
torch.load = _torch_load_patched

DATAROOT     = Path(r"E:\datasets\nuscenes")
# Use _datalist PKL — this is what CustomNuScenesDataset expects
SINGAPORE_PKL = DATAROOT / "nuscenes_infos_temporal_val_singapore_datalist.pkl"
OUTPUT_PKL   = Path(r"E:\bev_research\data\singapore_pseudo_labels.pkl")


def extract_preds_from_datasample(result):
    """Extract boxes, scores, labels from a Det3DDataSample result object."""
    if hasattr(result, "pred_instances_3d"):
        pred   = result.pred_instances_3d
        scores = pred.scores_3d.cpu().numpy()
        boxes  = pred.bboxes_3d.tensor.cpu().numpy()
        labels = pred.labels_3d.cpu().numpy()
    else:
        scores = np.array([])
        boxes  = np.zeros((0, 9), dtype=np.float32)
        labels = np.array([])
    return scores, boxes, labels


def main(args):
    from mmengine.config import Config
    from mmdet3d.apis    import init_model
    from mmdet3d.utils   import register_all_modules
    from mmdet3d.registry import DATASETS
    from mmengine.utils  import import_modules_from_strings
    import copy

    register_all_modules()

    print(f"Loading model:  {args.checkpoint}")
    cfg   = Config.fromfile(args.config)
    import_modules_from_strings(cfg.custom_imports.imports)
    model = init_model(cfg, args.checkpoint, device="cuda:0")
    model.eval()

    # Build dataset with Singapore datalist PKL
    val_cfg = copy.deepcopy(cfg.val_dataloader)
    val_cfg["dataset"]["ann_file"] = args.singapore_pkl
    dataset = DATASETS.build(val_cfg["dataset"])
    print(f"Total Singapore frames in dataset: {len(dataset)}")

    pseudo_labels = []
    score_stats   = []
    n_total       = 0
    n_kept        = 0

    print(f"\nRunning inference (score threshold = {args.score_threshold})...\n")
    with torch.no_grad():
        for frame_idx in range(len(dataset)):
            if frame_idx % 100 == 0:
                print(f"  [{frame_idx}/{len(dataset)}] processed...")
            try:
                sample = dataset[frame_idx]
                batched = {
                    "inputs":       {"img": sample["inputs"]["img"].unsqueeze(0).cuda()},
                    "data_samples": [sample["data_samples"]],
                }
                result = model.test_step(batched)
                scores, boxes, labels = extract_preds_from_datasample(result[0])

                n_total += len(scores)
                mask     = scores >= args.score_threshold
                n_kept  += int(mask.sum())
                score_stats.extend(scores.tolist())

                token = getattr(sample["data_samples"], "token",
                                f"frame_{frame_idx}")

                pseudo_info = {
                    "token":     token,
                    "gt_boxes":  boxes[mask].astype(np.float32),
                    "gt_labels": labels[mask].astype(np.int64),
                    "gt_scores": scores[mask].astype(np.float32),
                    "is_pseudo": True,
                }
                pseudo_labels.append(pseudo_info)

            except Exception as e:
                print(f"  [WARN] Frame {frame_idx}: {e}")

    print(f"\n{'='*52}")
    print(f"Frames processed:  {len(pseudo_labels)}")
    print(f"Total predictions: {n_total}")
    print(f"Above threshold:   {n_kept}  ({100*n_kept/max(n_total,1):.1f}%)")
    arr = np.array(score_stats) if score_stats else np.array([0.0])
    print(f"Score distribution:")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"  >= {thr:.1f}: {(arr >= thr).sum():5d} / {len(arr)}")

    out_pkl = Path(args.output)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    out = {"infos": pseudo_labels,
           "metadata": {"score_threshold": args.score_threshold}}
    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)
    print(f"\nPseudo labels saved -> {out_pkl}")

    stats = {
        "n_frames":            len(pseudo_labels),
        "n_total_predictions": int(n_total),
        "n_kept":              int(n_kept),
        "threshold":           args.score_threshold,
        "score_percentiles":   {str(p): float(np.percentile(arr, p))
                                for p in [10, 25, 50, 75, 90, 95]},
    }
    stats_path = out_pkl.with_suffix(".json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved -> {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          required=True)
    parser.add_argument("--checkpoint",      default=str(DATAROOT.parent /
                        "bev_research/checkpoints/bevformer_base_epoch_24.pth"))
    parser.add_argument("--singapore_pkl",   default=str(SINGAPORE_PKL))
    parser.add_argument("--output",          default=str(OUTPUT_PKL))
    parser.add_argument("--score_threshold", type=float, default=0.3)
    args = parser.parse_args()
    main(args)
