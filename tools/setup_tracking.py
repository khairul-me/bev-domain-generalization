"""
setup_tracking.py — Task 4.5: Initialize experiment tracking.
Run once to set up WandB and TensorBoard for the project.
"""
import subprocess
import sys
import os

PROJECT_ROOT = "E:/bev_research"
TENSORBOARD_DIR = f"{PROJECT_ROOT}/experiments/baseline/tensorboard_logs"

def setup_tensorboard():
    """Create TensorBoard log directory."""
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    print(f"[OK] TensorBoard log dir: {TENSORBOARD_DIR}")
    print(f"     To view: tensorboard --logdir={TENSORBOARD_DIR}")

def setup_wandb_offline():
    """Configure WandB in offline mode (no API key required initially)."""
    try:
        import wandb
        # Set to offline mode — sync later with 'wandb sync'
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_PROJECT'] = 'bev-domain-generalization'
        os.environ['WANDB_DIR'] = f"{PROJECT_ROOT}/experiments"
        
        print(f"[OK] WandB configured (offline mode)")
        print(f"     Project: bev-domain-generalization")
        print(f"     To sync online later: wandb sync {PROJECT_ROOT}/experiments/wandb/")
        print(f"     To use online: set WANDB_MODE=online and run: wandb login")
    except ImportError:
        print("[FAIL] WandB not installed")

def create_wandb_config():
    """Create a WandB config file for the project."""
    config = {
        "project": "bev-domain-generalization",
        "entity": None,  # Set to your WandB username
        "tags": ["bev-detection", "domain-adaptation", "depth-anything-v2"],
        "notes": "Domain-Generalizable Camera-Only BEV 3D Detection via Foundation Depth Priors",
        "config": {
            "model": "BEVFormer-Tiny-fp16",
            "backbone": "ResNet-50",
            "depth_prior": "Depth-Anything-V2-ViT-S",
            "adapter_channels": 256,
            "dataset_train": "nuScenes-v1.0-trainval",
            "dataset_eval": ["nuScenes-val", "KITTI-3D"],
            "total_epochs": 24,
            "batch_size": 1,
            "lr": 2e-4,
            "fp16": True,
        }
    }
    import json
    config_path = f"{PROJECT_ROOT}/configs/wandb_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[OK] WandB config saved: {config_path}")

if __name__ == "__main__":
    print("="*60)
    print("TASK 4.5: EXPERIMENT TRACKING SETUP")
    print("="*60)
    setup_tensorboard()
    setup_wandb_offline()
    create_wandb_config()
    print("\n[DONE] Experiment tracking configured.")
    print("       TensorBoard and WandB (offline) ready for training.")
