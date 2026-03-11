"""
run_evaluation.py — Tasks 10-11: Full evaluation pipeline.

Evaluates all 4 configurations (A-D) on nuScenes val and KITTI cross-domain.
Also supports ablation configs (E-H) for Task 12.

Config matrix:
  A = BEVFormer-Tiny-fp16 (original baseline)
  B = BEVFormer + DAv2 features (no adapter trained, random init)
  C = BEVFormer + DAv2 features + adapter (no TTA)
  D = BEVFormer + DAv2 features + adapter + TTA (full method)
  E = BEVFormer + depth VALUES (not features) from DAv2
  F = BEVFormer + DAv2 WITHOUT intrinsics normalization
  G = BEVFormer + adapter with random init (no DAv2)
  H = Full method but DAv2-Large (ViT-L)

Usage (after completing training):
  # Config A (baseline):
  python run_evaluation.py --config A --dataset nuscenes
  
  # Config D with TTA:
  python run_evaluation.py --config D --dataset kitti --use-tta
"""
import sys
import os
import json
import argparse

sys.path.insert(0, 'E:/Auto_Image/BEVFormer')
sys.path.insert(0, 'E:/bev_research/models')

EXPERIMENT_DIR = "E:/bev_research/experiments"
CHECKPOINT_DIR = "E:/bev_research/checkpoints"

CONFIG_MATRIX = {
    'A': {
        'description': 'BEVFormer-Tiny-fp16 (baseline)',
        'mmdet_config': 'E:/bev_research/configs/bevformer_rtx5060.py',
        'checkpoint': f'{CHECKPOINT_DIR}/bevformer_tiny_fp16_epoch_24.pth',
        'use_depth_prior': False,
        'use_intrinsics_norm': False,
        'use_tta': False,
    },
    'B': {
        'description': 'BEVFormer + DAv2 (random adapter, no training)',
        'mmdet_config': 'E:/bev_research/configs/bevformer_depth_prior_finetune.py',
        'checkpoint': f'{CHECKPOINT_DIR}/bevformer_tiny_fp16_epoch_24.pth',
        'use_depth_prior': True,
        'use_intrinsics_norm': True,
        'use_tta': False,
        'adapter_trained': False,
    },
    'C': {
        'description': 'BEVFormer + DAv2 + Trained Adapter (no TTA)',
        'mmdet_config': 'E:/bev_research/configs/bevformer_depth_prior_finetune.py',
        'checkpoint': f'{EXPERIMENT_DIR}/with_adapter/full_train/best.pth',
        'use_depth_prior': True,
        'use_intrinsics_norm': True,
        'use_tta': False,
    },
    'D': {
        'description': 'Ours: BEVFormer + DAv2 + Adapter + TTA (full method)',
        'mmdet_config': 'E:/bev_research/configs/bevformer_depth_prior_finetune.py',
        'checkpoint': f'{EXPERIMENT_DIR}/with_adapter/full_train/best.pth',
        'use_depth_prior': True,
        'use_intrinsics_norm': True,
        'use_tta': True,
        'tta_lr': 1e-4,       # From Task 9.5 sweep
        'tta_steps': 1,        # From Task 9.5 sweep
    },
    # Ablation configs
    'E': {
        'description': 'Ablation: Depth values (not features) from DAv2',
        'note': 'Use DAv2 inferred depth maps as input channel, not encoder features',
    },
    'F': {
        'description': 'Ablation: DAv2 without intrinsics normalization',
        'use_intrinsics_norm': False,
    },
    'G': {
        'description': 'Ablation: Random adapter (no DAv2 encoder)',
        'use_depth_prior': False,
        'note': 'Adapter receives random features, tests if structure alone helps',
    },
    'H': {
        'description': 'Ablation: DAv2-Large (ViT-L) backbone',
        'dav2_encoder': 'vitl',
        'note': 'Tests sensitivity to DAv2 model size',
    },
}


def run_nuscenes_evaluation(config_key, out_dir, use_tta=False):
    """
    Run nuScenes validation evaluation for a given config.
    
    Command:
        cd E:\Auto_Image\BEVFormer
        python tools/test.py {config} {checkpoint} --eval bbox \
            --out {out_dir}/results_nuscenes_val.pkl
    """
    cfg = CONFIG_MATRIX.get(config_key, {})
    mmdet_cfg = cfg.get('mmdet_config', '')
    ckpt = cfg.get('checkpoint', '')
    
    if not os.path.exists(ckpt):
        print(f"[{config_key}] Checkpoint not found: {ckpt} — skip (run training first)")
        return None
    
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = (
        f'cd E:\\Auto_Image\\BEVFormer && '
        f'conda run -n bev_research python tools/test.py '
        f'"{mmdet_cfg}" "{ckpt}" '
        f'--eval bbox '
        f'--out "{out_dir}\\results_nuscenes_val.pkl"'
    )
    print(f"[{config_key}] Command:\n  {cmd}")
    return cmd


def run_kitti_evaluation(config_key, out_dir):
    """
    Run KITTI cross-domain evaluation via adapted test script.
    """
    cfg = CONFIG_MATRIX.get(config_key, {})
    ckpt = cfg.get('checkpoint', '')
    
    if not os.path.exists(ckpt):
        print(f"[{config_key}] Checkpoint not found: {ckpt} — skip")
        return None
    
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = (
        f'conda run -n bev_research python '
        f'"E:\\bev_research\\tools\\test_cross_domain.py" '
        f'--checkpoint "{ckpt}" '
        f'--dataset kitti '
        f'--out "{out_dir}\\results_kitti_cross_domain.pkl"'
    )
    print(f"[{config_key}] Cross-domain command:\n  {cmd}")
    return cmd


def populate_results_table(results_dir):
    """
    Read all results PKL files and populate the results table.
    """
    print("\n" + "="*60)
    print("RESULTS TABLE (Tasks 10-11)")
    print("="*60)
    
    header = f"{'Config':<5} | {'Method':<45} | {'nuScenes NDS':>12} | {'nuScenes mAP':>12} | {'KITTI AP/E':>10} | {'Gen.Drop':>9}"
    print(header)
    print("-" * len(header))
    
    for cfg_key, cfg in CONFIG_MATRIX.items():
        if 'Ablation' in cfg.get('description', ''):
            continue  # Skip ablation configs for main table
        
        ns_nds = ns_map = kitti_ap = gen_drop = "TBD"
        
        ns_file = os.path.join(results_dir, cfg_key, "results_nuscenes_val.pkl")
        if os.path.exists(ns_file):
            # Load nuScenes results
            pass  # Populated after actual eval
        
        kitti_file = os.path.join(results_dir, cfg_key, "results_kitti_cross_domain.pkl")
        if os.path.exists(kitti_file):
            pass  # Populated after actual eval
        
        name = cfg.get('description', '')[:45]
        print(f"{cfg_key:<5} | {name:<45} | {ns_nds:>12} | {ns_map:>12} | {kitti_ap:>10} | {gen_drop:>9}")
    
    print("="*60)
    print("[TBD] = Results pending dataset download and training completion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=list(CONFIG_MATRIX.keys()), help='Config to run')
    parser.add_argument('--dataset', choices=['nuscenes', 'kitti', 'both'], default='both')
    parser.add_argument('--use-tta', action='store_true')
    parser.add_argument('--list-commands', action='store_true', help='Print all evaluation commands')
    args = parser.parse_args()
    
    if args.list_commands or args.config is None:
        print("="*60)
        print("ALL EVALUATION COMMANDS (run after training)")
        print("="*60)
        for cfg_key in ['A', 'B', 'C', 'D']:
            out_dir = f"{EXPERIMENT_DIR}/{cfg_key}"
            print(f"\n--- Config {cfg_key}: {CONFIG_MATRIX[cfg_key]['description']} ---")
            run_nuscenes_evaluation(cfg_key, out_dir)
            run_kitti_evaluation(cfg_key, out_dir)
        
        populate_results_table(EXPERIMENT_DIR)
    elif args.config:
        out_dir = f"{EXPERIMENT_DIR}/{args.config}"
        if args.dataset in ('nuscenes', 'both'):
            run_nuscenes_evaluation(args.config, out_dir, args.use_tta)
        if args.dataset in ('kitti', 'both'):
            run_kitti_evaluation(args.config, out_dir)
