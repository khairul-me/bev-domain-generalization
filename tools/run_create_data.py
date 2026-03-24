#!/usr/bin/env python
"""
run_create_data.py — Run dataset pkl generation (Fix 4).
Uses BEVFormer's create_data for temporal format (nuScenes) and KITTI.
Run from E:\bev_research with: python tools/run_create_data.py [nuscenes|kitti|all]
"""
import sys
import os
import subprocess

sys.path.insert(0, 'E:/bev_research')
from config.paths import BEVFORMER_ROOT, NUSCENES_ROOT, KITTI_ROOT

def run_nuscenes():
    """Generate nuscenes_infos_temporal_*.pkl"""
    cmd = [
        sys.executable,
        "tools/create_data.py", "nuscenes",
        "--root-path", NUSCENES_ROOT,
        "--out-dir", NUSCENES_ROOT,
        "--canbus", NUSCENES_ROOT,
        "--extra-tag", "nuscenes",
        "--version", "v1.0"
    ]
    return subprocess.call(cmd, cwd=BEVFORMER_ROOT)

def run_kitti():
    """Generate kitti_infos_*.pkl"""
    cmd = [
        sys.executable,
        "tools/create_data.py", "kitti",
        "--root-path", KITTI_ROOT,
        "--out-dir", KITTI_ROOT,
        "--extra-tag", "kitti"
    ]
    return subprocess.call(cmd, cwd=BEVFORMER_ROOT)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "nuscenes":
        sys.exit(run_nuscenes())
    elif target == "kitti":
        sys.exit(run_kitti())
    elif target == "all":
        if run_nuscenes() != 0:
            sys.exit(1)
        if run_kitti() != 0:
            sys.exit(1)
        print("All pkl files generated successfully.")
    else:
        print("Usage: python run_create_data.py [nuscenes|kitti|all]")
        sys.exit(1)
