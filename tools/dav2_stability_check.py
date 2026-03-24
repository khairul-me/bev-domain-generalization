import argparse
import json
import os
import pickle
import random
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


def _load_infos(pkl_path: str) -> List[dict]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        infos = data["infos"]
    elif isinstance(data, dict) and "data_list" in data:
        infos = data["data_list"]
    else:
        raise ValueError(f"Unsupported PKL format: {pkl_path}")
    return list(sorted(infos, key=lambda x: x["timestamp"]))


def _resolve_img_path(data_root: str, img_path: str) -> str:
    img_path = img_path.replace("\\", "/")
    if os.path.isabs(img_path):
        return img_path
    return os.path.join(data_root, img_path)


def _extract_front_paths(infos: List[dict], data_root: str) -> List[str]:
    paths = []
    for info in infos:
        cams = info.get("cams", {})
        if "CAM_FRONT" not in cams:
            continue
        p = cams["CAM_FRONT"].get("data_path")
        if not p:
            continue
        paths.append(_resolve_img_path(data_root, p))
    return paths


def _sample_paths(paths: List[str], n: int, seed: int) -> List[str]:
    if len(paths) <= n:
        return paths
    rng = random.Random(seed)
    idxs = list(range(len(paths)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:n])
    return [paths[i] for i in idxs]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if a.size != b.size or a.size == 0:
        return float("nan")
    sa = np.std(a)
    sb = np.std(b)
    if sa < 1e-8 or sb < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2) + 1e-12)
    return float((np.mean(a) - np.mean(b)) / pooled)


def _depth_features(depth: np.ndarray, raw_bgr: np.ndarray) -> Dict[str, float]:
    d = np.clip(depth.astype(np.float32), 1e-6, None)
    logd = np.log(d)
    logd = (logd - np.mean(logd)) / (np.std(logd) + 1e-6)

    gx = cv2.Sobel(logd, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(logd, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    igx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    igy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    img_grad = np.sqrt(igx * igx + igy * igy)

    p05 = float(np.percentile(d, 5))
    p50 = float(np.percentile(d, 50))
    p95 = float(np.percentile(d, 95))

    return {
        "log_std": float(np.std(logd)),
        "grad_mean": float(np.mean(grad_mag)),
        "grad_p90": float(np.percentile(grad_mag, 90)),
        "edge_corr": _safe_corr(grad_mag, img_grad),
        "p95_over_p50": float(p95 / (p50 + 1e-6)),
        "p50_over_p05": float(p50 / (p05 + 1e-6)),
    }


def _aggregate(stats: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = list(stats[0].keys())
    out = {}
    for k in keys:
        vals = np.array([x[k] for x in stats], dtype=np.float64)
        out[k] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="DAv2 Boston/Singapore stability check.")
    parser.add_argument("--data-root", default="C:/datasets/nuscenes")
    parser.add_argument("--boston-pkl", default="C:/datasets/nuscenes/nuscenes_infos_temporal_val_boston.pkl")
    parser.add_argument("--singapore-pkl", default="C:/datasets/nuscenes/nuscenes_infos_temporal_val_singapore.pkl")
    parser.add_argument("--depth-repo", default="E:/Auto_Image/bev_research/Depth-Anything-V2")
    parser.add_argument("--ckpt", default="E:/bev_research/checkpoints/depth_anything_v2_vits.pth")
    parser.add_argument("--samples-per-domain", type=int, default=50)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", default="E:/bev_research/logs/dav2_stability_100frames.json")
    args = parser.parse_args()

    sys.path.insert(0, args.depth_repo)
    from depth_anything_v2.dpt import DepthAnythingV2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.to(device).eval()

    boston_infos = _load_infos(args.boston_pkl)
    singapore_infos = _load_infos(args.singapore_pkl)
    boston_paths = _sample_paths(_extract_front_paths(boston_infos, args.data_root), args.samples_per_domain, args.seed)
    singapore_paths = _sample_paths(
        _extract_front_paths(singapore_infos, args.data_root), args.samples_per_domain, args.seed + 1
    )

    if len(boston_paths) == 0 or len(singapore_paths) == 0:
        raise RuntimeError("No CAM_FRONT image paths found in one or both domains.")

    results: Dict[str, List[Dict[str, float]]] = {"boston": [], "singapore": []}

    for domain, paths in [("boston", boston_paths), ("singapore", singapore_paths)]:
        for i, p in enumerate(paths, start=1):
            img = cv2.imread(p)
            if img is None:
                continue
            depth = model.infer_image(img, input_size=args.input_size)
            feats = _depth_features(depth, img)
            results[domain].append(feats)
            if i % 10 == 0:
                print(f"[{domain}] processed {i}/{len(paths)}")

    if len(results["boston"]) < 10 or len(results["singapore"]) < 10:
        raise RuntimeError("Too few valid images processed. Check image paths/data root.")

    agg = {
        "boston": _aggregate(results["boston"]),
        "singapore": _aggregate(results["singapore"]),
    }

    comparison = {}
    for key in agg["boston"].keys():
        b = np.array([x[key] for x in results["boston"]], dtype=np.float64)
        s = np.array([x[key] for x in results["singapore"]], dtype=np.float64)
        comparison[key] = {
            "delta_mean_sg_minus_boston": float(np.nanmean(s) - np.nanmean(b)),
            "cohens_d_sg_vs_boston": _cohens_d(s, b),
        }

    report = {
        "config": {
            "samples_per_domain": args.samples_per_domain,
            "total_frames": args.samples_per_domain * 2,
            "camera": "CAM_FRONT",
            "encoder": "DAv2 ViT-S",
            "input_size": args.input_size,
            "seed": args.seed,
        },
        "counts": {
            "boston_processed": len(results["boston"]),
            "singapore_processed": len(results["singapore"]),
        },
        "aggregate": agg,
        "comparison": comparison,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved report to: {args.out_json}")


if __name__ == "__main__":
    main()
