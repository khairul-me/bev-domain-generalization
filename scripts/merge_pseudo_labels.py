"""
merge_pseudo_labels.py

Merge Singapore pseudo-labels into the original Singapore datalist PKL.

The pseudo-labels PKL only has (token, gt_boxes, gt_labels, gt_scores).
CustomNuScenesDataset requires the full frame metadata (cams, timestamp,
calibration, etc.) from the original PKL.

This script:
  1. Loads the original Singapore datalist PKL (full metadata)
  2. Loads the pseudo-labels PKL (prediction boxes above threshold)
  3. For each frame, replaces gt_boxes/gt_labels with pseudo-labels
  4. Saves a merged PKL that can be used directly as ann_file for E5 training

Usage:
    python merge_pseudo_labels.py
"""

import pickle, sys
from pathlib import Path

DATAROOT        = Path(r"E:\datasets\nuscenes")
SING_DATALIST   = DATAROOT / "nuscenes_infos_temporal_val_singapore_datalist.pkl"
PSEUDO_PKL      = Path(r"E:\bev_research\data\singapore_pseudo_labels.pkl")
OUTPUT_PKL      = Path(r"E:\bev_research\data\singapore_pseudo_labels_merged.pkl")

sys.path.insert(0, r"E:\Auto_Image\bev_research\mmdetection3d")


def load_infos(pkl_path):
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        return raw.get("data_list", raw.get("infos", [])), raw
    return raw, None


def main():
    print(f"Loading Singapore datalist: {SING_DATALIST}")
    sing_infos, sing_raw = load_infos(SING_DATALIST)
    print(f"  {len(sing_infos)} frames in original datalist")

    print(f"Loading pseudo-labels: {PSEUDO_PKL}")
    pseudo_infos, _ = load_infos(PSEUDO_PKL)
    print(f"  {len(pseudo_infos)} frames with pseudo-labels")

    # nuScenes class names in canonical order (indices 0-9)
    NUSCENES_CLASSES = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
    ]

    # Both PKLs come from the same Singapore datalist in the same order.
    # Merge by sequential index (positional alignment) since tokens in
    # pseudo-labels were saved as "frame_N" fallbacks.
    print("Merging pseudo-labels into original infos (sequential index)...")
    import copy, numpy as np
    merged = []
    n_matched = 0
    for i, (info, pl) in enumerate(zip(sing_infos, pseudo_infos)):
        new_info = copy.deepcopy(info)
        n_boxes   = len(pl["gt_boxes"])
        gt_labels = pl["gt_labels"]

        new_info["gt_boxes"]  = pl["gt_boxes"][:, :7].astype(np.float32)   # x,y,z,l,w,h,yaw only
        new_info["gt_labels"] = gt_labels
        new_info["gt_scores"] = pl.get("gt_scores", np.zeros(n_boxes, dtype="float32"))
        new_info["is_pseudo"] = True

        # gt_names must align with gt_boxes so get_ann_info mask filtering works
        new_info["gt_names"] = np.array(
            [NUSCENES_CLASSES[lbl] if 0 <= lbl < len(NUSCENES_CLASSES) else 'ignore'
             for lbl in gt_labels], dtype=object
        )
        # valid_flag / num_lidar_pts must also align with gt_boxes length
        new_info["valid_flag"]     = np.ones(n_boxes, dtype=bool)
        new_info["num_lidar_pts"]  = np.ones(n_boxes, dtype=np.int32)

        # gt_velocity: split from pseudo-label 9D boxes (cols 7-8 are vx, vy)
        if pl["gt_boxes"].shape[1] >= 9:
            new_info["gt_velocity"] = pl["gt_boxes"][:, 7:9].astype(np.float32)
        else:
            new_info["gt_velocity"] = np.zeros((n_boxes, 2), dtype=np.float32)

        merged.append(new_info)
        n_matched += 1

    print(f"  Merged {n_matched} frames")

    # Build output dict matching original format
    if isinstance(sing_raw, dict) and "data_list" in sing_raw:
        out = dict(sing_raw)
        out["data_list"] = merged
        out["metadata"] = {"pseudo_label": True, "score_threshold": 0.3}
    elif isinstance(sing_raw, dict) and "infos" in sing_raw:
        out = dict(sing_raw)
        out["infos"] = merged
        out["metadata"] = {"pseudo_label": True, "score_threshold": 0.3}
    else:
        out = {"infos": merged, "metadata": {"pseudo_label": True}}

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(out, f)
    print(f"\nMerged PKL saved -> {OUTPUT_PKL}")
    print(f"  Frames: {len(merged)}")
    total_boxes = sum(len(i["gt_boxes"]) for i in merged)
    print(f"  Total pseudo-label boxes: {total_boxes}")


if __name__ == "__main__":
    main()
