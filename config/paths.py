# E:\bev_research\config\paths.py
# ============================================================
# Central path configuration for the entire project.
# All scripts should import paths from here.
# If your folder structure changes, update ONLY this file.
# ============================================================

import os

# --- Core roots ---
PROJECT_ROOT   = "E:/bev_research"
BEVFORMER_ROOT = "E:/Auto_Image/BEVFormer"
MMDET3D_ROOT   = "E:/Auto_Image/bev_research/mmdetection3d"
DAV2_ROOT      = "E:/Auto_Image/Depth-Anything-V2"

# --- Dataset paths ---
DATASET_ROOT   = "C:/datasets"
NUSCENES_ROOT  = os.path.join(DATASET_ROOT, "nuscenes")
KITTI_ROOT     = os.path.join(DATASET_ROOT, "kitti")

# --- pkl info files (update these after Fix 4 if names differ) ---
NUSCENES_TRAIN_PKL = os.path.join(NUSCENES_ROOT, "nuscenes_infos_temporal_train.pkl")
NUSCENES_VAL_PKL   = os.path.join(NUSCENES_ROOT, "nuscenes_infos_temporal_val.pkl")
KITTI_TRAIN_PKL    = os.path.join(KITTI_ROOT, "kitti_infos_train.pkl")
KITTI_VAL_PKL      = os.path.join(KITTI_ROOT, "kitti_infos_val.pkl")

# --- Experiment outputs (keep on E: drive for space) ---
EXPERIMENTS_ROOT   = os.path.join(PROJECT_ROOT, "experiments")
CHECKPOINTS_ROOT   = os.path.join(PROJECT_ROOT, "checkpoints")
LOGS_ROOT          = os.path.join(PROJECT_ROOT, "logs")

# --- Python interpreter ---
PYTHON_EXE = r"C:\Users\Khairul\miniconda3\envs\bev310\python.exe"
