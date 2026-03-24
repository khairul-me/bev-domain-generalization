import argparse
import json
import os
import pickle
import random
import sys
from typing import Dict, List, Tuple

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from nuscenes.nuscenes import NuScenes


def load_infos(pkl_path: str) -> List[dict]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        infos = data["infos"]
    elif isinstance(data, dict) and "data_list" in data:
        infos = data["data_list"]
    else:
        raise ValueError(f"Unsupported PKL format: {pkl_path}")
    return list(sorted(infos, key=lambda x: x["timestamp"]))


def resolve_img_path(data_root: str, img_path: str) -> str:
    norm = img_path.replace("\\", "/")
    if os.path.isabs(norm):
        return norm
    return os.path.join(data_root, norm)


def build_scene_desc_map(nusc: NuScenes) -> Dict[str, str]:
    out = {}
    for scene in nusc.scene:
        out[scene["token"]] = scene.get("description", "")
    return out


def build_candidates(
    infos: List[dict], data_root: str, scene_desc_map: Dict[str, str], domain: str
) -> List[dict]:
    candidates = []
    for info in infos:
        cams = info.get("cams", {})
        if "CAM_FRONT" not in cams:
            continue
        img_rel = cams["CAM_FRONT"].get("data_path")
        if not img_rel:
            continue
        img_path = resolve_img_path(data_root, img_rel)
        scene_desc = scene_desc_map.get(info.get("scene_token", ""), "").lower()
        is_night = "night" in scene_desc
        candidates.append(
            {
                "domain": domain,
                "token": info["token"],
                "scene_token": info.get("scene_token", ""),
                "scene_desc": scene_desc,
                "is_night": is_night,
                "img_path": img_path,
            }
        )
    return candidates


def sample_boston(candidates: List[dict], n: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    idx = list(range(len(candidates)))
    rng.shuffle(idx)
    idx = sorted(idx[: min(n, len(candidates))])
    return [candidates[i] for i in idx]


def sample_singapore_with_night(candidates: List[dict], n: int, seed: int, min_night: int) -> List[dict]:
    rng = random.Random(seed)
    night = [c for c in candidates if c["is_night"]]
    day = [c for c in candidates if not c["is_night"]]
    rng.shuffle(night)
    rng.shuffle(day)

    take_night = min(min_night, len(night), n)
    selected = night[:take_night]
    rem = n - len(selected)
    if rem > 0:
        selected.extend(day[:rem])
        rem2 = n - len(selected)
        if rem2 > 0:
            selected.extend(night[take_night : take_night + rem2])
    return selected[:n]


def load_model(depth_repo: str, ckpt_path: str):
    sys.path.insert(0, depth_repo)
    from depth_anything_v2.dpt import DepthAnythingV2

    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()
    return model


def infer_depth(model, img_bgr: np.ndarray, input_size: int) -> np.ndarray:
    return model.infer_image(img_bgr, input_size=input_size).astype(np.float32)


def normalize_depth_vis(depth: np.ndarray) -> np.ndarray:
    d = depth.copy()
    d = (d - np.min(d)) / (np.max(d) - np.min(d) + 1e-6)
    d = (d * 255.0).astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    colored = (cmap(d)[:, :, :3] * 255).astype(np.uint8)
    return colored[:, :, ::-1]  # RGB->BGR


def collect_depth_values(items: List[dict], model, input_size: int) -> Tuple[List[dict], np.ndarray]:
    processed = []
    all_vals = []
    for i, item in enumerate(items, start=1):
        img = cv2.imread(item["img_path"])
        if img is None:
            continue
        depth = infer_depth(model, img, input_size=input_size)
        processed.append({**item, "img": img, "depth": depth})
        # Use robust clipping for histogram stability.
        vals = np.clip(depth.flatten(), 1e-6, np.percentile(depth, 99.5))
        all_vals.append(vals)
        if i % 10 == 0:
            print(f"[{item['domain']}] processed {i}/{len(items)}")
    if len(all_vals) == 0:
        return processed, np.array([], dtype=np.float32)
    return processed, np.concatenate(all_vals, axis=0)


def save_histogram(boston_vals: np.ndarray, singapore_vals: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    bins = 120
    plt.hist(boston_vals, bins=bins, density=True, alpha=0.5, label="Boston")
    plt.hist(singapore_vals, bins=bins, density=True, alpha=0.5, label="Singapore")
    plt.xlabel("DAv2 depth value (relative)")
    plt.ylabel("Density")
    plt.title("DAv2 Depth Distribution: Boston vs Singapore (100 frames)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_log_histogram(boston_vals: np.ndarray, singapore_vals: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    bins = 120
    b = np.log(np.clip(boston_vals, 1e-6, None))
    s = np.log(np.clip(singapore_vals, 1e-6, None))
    plt.hist(b, bins=bins, density=True, alpha=0.5, label="Boston")
    plt.hist(s, bins=bins, density=True, alpha=0.5, label="Singapore")
    plt.xlabel("log(depth)")
    plt.ylabel("Density")
    plt.title("DAv2 log-depth Distribution: Boston vs Singapore")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_visual_pairs(
    boston_items: List[dict], singapore_night_items: List[dict], out_dir: str, n_pairs: int
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    n = min(n_pairs, len(boston_items), len(singapore_night_items))
    for i in range(n):
        b = boston_items[i]
        s = singapore_night_items[i]
        b_vis = normalize_depth_vis(b["depth"])
        s_vis = normalize_depth_vis(s["depth"])
        b_panel = cv2.hconcat([b["img"], b_vis])
        s_panel = cv2.hconcat([s["img"], s_vis])
        canvas = cv2.vconcat([b_panel, s_panel])
        text1 = "Top: Boston (day likely), Bottom: Singapore night"
        text2 = f"boston_token={b['token'][:8]}  singapore_token={s['token'][:8]}"
        cv2.putText(canvas, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, text2, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        out = os.path.join(out_dir, f"pair_{i+1:02d}.png")
        cv2.imwrite(out, canvas)
        saved.append(out)
    return saved


def summarize_vals(vals: np.ndarray) -> Dict[str, float]:
    vals = vals.astype(np.float64)
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
    }


def overlap_coeff(a: np.ndarray, b: np.ndarray, bins: int = 200) -> float:
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if hi <= lo:
        return 1.0
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    pa = ha / max(np.sum(ha), 1)
    pb = hb / max(np.sum(hb), 1)
    return float(np.sum(np.minimum(pa, pb)))


def main():
    parser = argparse.ArgumentParser(description="DAv2 stability gate pack (100 frames + visuals).")
    parser.add_argument("--data-root", default="E:/datasets/nuscenes")
    parser.add_argument("--boston-pkl", default="C:/datasets/nuscenes/nuscenes_infos_temporal_val_boston.pkl")
    parser.add_argument("--singapore-pkl", default="C:/datasets/nuscenes/nuscenes_infos_temporal_val_singapore.pkl")
    parser.add_argument("--depth-repo", default="E:/Auto_Image/bev_research/Depth-Anything-V2")
    parser.add_argument("--ckpt", default="E:/bev_research/checkpoints/depth_anything_v2_vits.pth")
    parser.add_argument("--samples-per-domain", type=int, default=50)
    parser.add_argument("--min-singapore-night", type=int, default=25)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="E:/bev_research/analysis/dav2_stability_gate")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pairs_dir = os.path.join(args.out_dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.data_root, verbose=False)
    scene_desc_map = build_scene_desc_map(nusc)

    boston_infos = load_infos(args.boston_pkl)
    singapore_infos = load_infos(args.singapore_pkl)
    boston_candidates = build_candidates(boston_infos, args.data_root, scene_desc_map, "boston")
    singapore_candidates = build_candidates(singapore_infos, args.data_root, scene_desc_map, "singapore")

    boston_sample = sample_boston(boston_candidates, args.samples_per_domain, args.seed)
    singapore_sample = sample_singapore_with_night(
        singapore_candidates, args.samples_per_domain, args.seed + 1, args.min_singapore_night
    )
    singapore_night_only = [x for x in singapore_sample if x["is_night"]]

    model = load_model(args.depth_repo, args.ckpt)
    boston_proc, boston_vals = collect_depth_values(boston_sample, model, args.input_size)
    singapore_proc, singapore_vals = collect_depth_values(singapore_sample, model, args.input_size)
    singapore_night_proc = [x for x in singapore_proc if x["is_night"]]
    singapore_night_vals = []
    for x in singapore_night_proc:
        d = x["depth"]
        singapore_night_vals.append(np.clip(d.flatten(), 1e-6, np.percentile(d, 99.5)))
    singapore_night_vals = (
        np.concatenate(singapore_night_vals, axis=0)
        if len(singapore_night_vals) > 0
        else np.array([], dtype=np.float32)
    )

    hist_path = os.path.join(args.out_dir, "hist_depth_boston_vs_singapore.png")
    log_hist_path = os.path.join(args.out_dir, "hist_logdepth_boston_vs_singapore.png")
    save_histogram(boston_vals, singapore_vals, hist_path)
    save_log_histogram(boston_vals, singapore_vals, log_hist_path)

    pair_paths = save_visual_pairs(boston_proc, singapore_night_proc, pairs_dir, n_pairs=10)

    summary = {
        "config": {
            "samples_per_domain": args.samples_per_domain,
            "min_singapore_night": args.min_singapore_night,
            "input_size": args.input_size,
            "seed": args.seed,
            "camera": "CAM_FRONT",
            "model": "Depth-Anything-V2 ViT-S (frozen)",
        },
        "counts": {
            "boston_candidates": len(boston_candidates),
            "singapore_candidates": len(singapore_candidates),
            "singapore_night_candidates": sum(1 for x in singapore_candidates if x["is_night"]),
            "boston_processed": len(boston_proc),
            "singapore_processed": len(singapore_proc),
            "singapore_night_processed": len(singapore_night_proc),
            "pair_count": len(pair_paths),
        },
        "artifacts": {
            "hist_depth": hist_path,
            "hist_logdepth": log_hist_path,
            "pairs_dir": pairs_dir,
            "pairs": pair_paths,
        },
        "stats": {
            "boston": summarize_vals(boston_vals),
            "singapore": summarize_vals(singapore_vals),
            "singapore_night": summarize_vals(singapore_night_vals)
            if singapore_night_vals.size > 0
            else None,
            "overlap_coeff": {
                "boston_vs_singapore": overlap_coeff(boston_vals, singapore_vals),
                "boston_vs_singapore_night": overlap_coeff(boston_vals, singapore_night_vals)
                if singapore_night_vals.size > 0
                else None,
            },
        },
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
