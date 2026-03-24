"""
create_data_skip_gt.py — Create pkl files WITHOUT create_groundtruth_database.
Use when mmcv.ops has DLL issues. GT database is for data augmentation; inference works without it.
Run: python tools/create_data_skip_gt.py [kitti|nuscenes] [args...]
"""
import sys
import os
import argparse
from pathlib import Path

# Add BEVFormer tools to path (data_converter is in tools/)
BEVFORMER = r"E:\Auto_Image\BEVFormer"
BEVFORMER_TOOLS = os.path.join(BEVFORMER, "tools")
sys.path.insert(0, BEVFORMER_TOOLS)
os.chdir(BEVFORMER)

# Import converters (do NOT import create_gt_database)
from data_converter import nuscenes_converter as nuscenes_converter
from data_converter import kitti_converter as kitti

def kitti_prep(root_path, out_dir, info_prefix='kitti'):
    """KITTI prep without GT database."""
    kitti.create_kitti_info_file(root_path, info_prefix)
    kitti.create_reduced_point_cloud(root_path, info_prefix)
    info_train = os.path.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val = os.path.join(root_path, f'{info_prefix}_infos_val.pkl')
    info_trainval = os.path.join(root_path, f'{info_prefix}_infos_trainval.pkl')
    info_test = os.path.join(root_path, f'{info_prefix}_infos_test.pkl')
    kitti.export_2d_annotation(root_path, info_train)
    kitti.export_2d_annotation(root_path, info_val)
    kitti.export_2d_annotation(root_path, info_trainval)
    kitti.export_2d_annotation(root_path, info_test)
    print("KITTI pkl files created (GT database skipped).")

def nuscenes_prep(root_path, out_dir, canbus, info_prefix='nuscenes', version='v1.0-trainval'):
    """nuScenes prep (no GT database - BEVFormer doesn't use it for nuScenes)."""
    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, canbus, info_prefix, version=version, max_sweeps=10)
    if 'test' in version:
        info_path = os.path.join(out_dir, f'{info_prefix}_infos_temporal_test.pkl')
        nuscenes_converter.export_2d_annotation(root_path, info_path, version=version)
    else:
        info_train = os.path.join(out_dir, f'{info_prefix}_infos_temporal_train.pkl')
        info_val = os.path.join(out_dir, f'{info_prefix}_infos_temporal_val.pkl')
        nuscenes_converter.export_2d_annotation(root_path, info_train, version=version)
        nuscenes_converter.export_2d_annotation(root_path, info_val, version=version)
    print("nuScenes pkl files created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['kitti', 'nuscenes'])
    parser.add_argument('--root-path', default='C:/datasets/kitti')
    parser.add_argument('--out-dir', default='C:/datasets/kitti')
    parser.add_argument('--canbus', default='C:/datasets/nuscenes')
    parser.add_argument('--extra-tag', default='kitti')
    parser.add_argument('--version', default='v1.0')
    args = parser.parse_args()

    if args.dataset == 'kitti':
        kitti_prep(args.root_path, args.out_dir, args.extra_tag)
    elif args.dataset == 'nuscenes':
        ver = f'{args.version}-trainval' if args.version == 'v1.0' else args.version
        nuscenes_prep(args.root_path, args.out_dir, args.canbus, args.extra_tag, ver)
