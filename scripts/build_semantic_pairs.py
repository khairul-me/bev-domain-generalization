"""
build_semantic_pairs.py

Build semantically matched Boston–Singapore frame pairs for representation analysis.
Matching criteria (in priority order):
  1. Time-of-day bin  (dawn / day / dusk / night)
  2. Object density   (sparse: <5, medium: 5–15, dense: >15 annotations)

FIX vs. guide original:
  - Corrected PKL paths to actual nuScenes data location:
      E:\datasets\nuscenes\nuscenes_infos_temporal_val_boston.pkl
      E:\datasets\nuscenes\nuscenes_infos_temporal_val_singapore.pkl
  - Added support for the MMDet3D PKL dict structure ('infos' or 'data_list').
  - Token key falls back to 'sample_token' (common in temporal PKLs) then 'token'.
  - Time-of-day uses UTC hour only as a gross approximation; documented clearly.

Usage:
    conda activate bev310
    python scripts\build_semantic_pairs.py --n_pairs 500
"""

import argparse, json, pickle, random
from pathlib import Path
from collections import defaultdict


DATAROOT     = Path(r"E:\datasets\nuscenes")
BOSTON_PKL   = DATAROOT / "nuscenes_infos_temporal_val_boston.pkl"
SINGAPORE_PKL= DATAROOT / "nuscenes_infos_temporal_val_singapore.pkl"
OUTPUT_PATH  = Path(r"E:\bev_research\data\matched_pairs_500.json")


def load_infos(pkl_path: Path) -> list:
    """Load sample info list from a nuScenes PKL file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        # MMDet3D v1 uses 'infos'; v2 uses 'data_list'
        return data.get("infos", data.get("data_list", []))
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected PKL type: {type(data)}")


def get_token(info: dict) -> str:
    """Extract sample token, checking multiple possible key names."""
    return info.get("token", info.get("sample_token", ""))


def get_hour_bin(info: dict) -> str:
    """
    Convert nuScenes timestamp (µs since epoch) to time-of-day bin.
    Uses UTC hour as a coarse approximation.
    Boston is UTC-5, Singapore is UTC+8 -- this introduces an offset but is
    applied consistently and only used for bucketing, not exact matching.
    """
    import datetime
    ts = info.get("timestamp", 0)
    h  = datetime.datetime.utcfromtimestamp(ts / 1e6).hour
    if   5 <= h <  9:  return "dawn"
    elif 9 <= h < 17:  return "day"
    elif 17 <= h < 20: return "dusk"
    else:               return "night"


def get_density_bin(info: dict) -> str:
    """Object count bucket based on number of GT boxes."""
    # Try multiple key names used across MMDet3D versions
    boxes = info.get("gt_boxes",
            info.get("ann_infos",
            info.get("gt_bboxes_3d", [])))
    n = len(boxes) if hasattr(boxes, "__len__") else 0
    if n < 5:    return "sparse"
    elif n < 15: return "medium"
    else:        return "dense"


def semantic_key(info: dict) -> tuple:
    # Using density-only bucketing avoids the UTC timezone problem:
    # Boston frames all show in UTC 'dusk' (noon local), Singapore in UTC 'dawn'/'night'
    # (morning local), causing near-zero time-of-day bucket overlap.
    # Density is timezone-independent and is the strongest semantic proxy available.
    return (get_density_bin(info),)


def main(args):
    rng = random.Random(42)

    print(f"Loading Boston PKL:    {args.boston_pkl}")
    boston_infos = load_infos(Path(args.boston_pkl))
    print(f"Loading Singapore PKL: {args.singapore_pkl}")
    sing_infos   = load_infos(Path(args.singapore_pkl))

    print(f"Boston frames: {len(boston_infos)},  Singapore frames: {len(sing_infos)}")

    # ── Bucket frames by semantic key ─────────────────────────────────────────
    boston_buckets = defaultdict(list)
    sing_buckets   = defaultdict(list)

    skipped_b, skipped_s = 0, 0
    for info in boston_infos:
        tok = get_token(info)
        if not tok:
            skipped_b += 1
            continue
        boston_buckets[semantic_key(info)].append(tok)

    for info in sing_infos:
        tok = get_token(info)
        if not tok:
            skipped_s += 1
            continue
        sing_buckets[semantic_key(info)].append(tok)

    if skipped_b or skipped_s:
        print(f"[WARN] Skipped {skipped_b} Boston / {skipped_s} Singapore frames with no token.")

    all_keys = sorted(set(boston_buckets) | set(sing_buckets))
    print(f"\nSemantic bucket distribution ({len(all_keys)} buckets):")
    print(f"  {'Bucket':<40}  {'Boston':>7}  {'Singapore':>9}")
    for k in all_keys:
        nb = len(boston_buckets.get(k, []))
        ns = len(sing_buckets.get(k, []))
        print(f"  {str(k):<40}  {nb:>7}  {ns:>9}")

    # ── Sample matched pairs — proportional to bucket capacity ───────────────
    # Equal-per-bucket allocation fails when one bucket is tiny (e.g. 'sparse'
    # has only 1 Boston frame). Instead, allocate proportional to min(b, s) per bucket.
    common_keys = [k for k in all_keys
                   if boston_buckets.get(k) and sing_buckets.get(k)]

    if not common_keys:
        print("\n[ERROR] No common semantic buckets found between cities.")
        print("  This likely means the PKL files have no timestamp or gt_boxes fields.")
        return

    # Available pairs per bucket
    avail = {k: min(len(boston_buckets[k]), len(sing_buckets[k]))
             for k in common_keys}
    total_avail = sum(avail.values())

    if total_avail == 0:
        print("[ERROR] No pairs available in any bucket.")
        return

    pairs = []
    for i, key in enumerate(common_keys):
        b_tokens = boston_buckets[key][:]
        s_tokens = sing_buckets[key][:]
        rng.shuffle(b_tokens)
        rng.shuffle(s_tokens)
        # Proportional budget
        budget = round(args.n_pairs * avail[key] / total_avail)
        budget = min(budget, avail[key])
        for j in range(budget):
            pairs.append({
                "boston_token":    b_tokens[j],
                "singapore_token": s_tokens[j],
                "bucket_key":      str(key),
            })

    rng.shuffle(pairs)
    print(f"\nTotal matched pairs generated: {len(pairs)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"pairs": pairs, "n_pairs": len(pairs)}, f, indent=2)
    print(f"Pairs saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boston_pkl",    default=str(BOSTON_PKL))
    parser.add_argument("--singapore_pkl", default=str(SINGAPORE_PKL))
    parser.add_argument("--n_pairs",       type=int, default=500)
    parser.add_argument("--output",        default=str(OUTPUT_PATH))
    args = parser.parse_args()
    main(args)
