"""
create_kitti_info_only.py — Create KITTI pkl files WITHOUT create_groundtruth_database.
Use when mmcv.ops has DLL issues. The GT database is for data augmentation; inference can work without it.
"""
import sys
import os
from pathlib import Path

# Add mmdetection3d
mmdet3d_root = r"E:\Auto_Image\bev_research\mmdetection3d"
sys.path.insert(0, mmdet3d_root)

from tools.dataset_converters import kitti_converter as kitti
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos
from os import path as osp

def main():
    root_path = "C:/datasets/kitti"
    out_dir = "C:/datasets/kitti"
    info_prefix = "kitti"
    
    print("Creating KITTI info files (skipping GT database)...")
    kitti.create_kitti_info_file(root_path, info_prefix, with_plane=False)
    
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    
    print("Updating pkl infos to v2...")
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_trainval_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_test_path)
    
    print("Done! Created:", info_train_path, info_val_path)
    print("Note: kitti_dbinfos_train.pkl was NOT created (GT database skipped).")
    print("Training with data augmentation may need it; inference/eval should work.")

if __name__ == "__main__":
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')
    main()
