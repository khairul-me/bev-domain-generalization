"""
Comprehensive PKL fixer: adds new-format calibration fields to datalist PKLs.

The NuScenesMetric in mmdet3d 1.4.0 expects each data_list entry to have:
  - info['token']                        (string)
  - info['lidar_points']['lidar2ego']    (4x4 numpy matrix)
  - info['ego2global']                   (4x4 numpy matrix)

Legacy BEVFormer PKLs only have:
  - info['lidar2ego_rotation']           (quaternion [w,x,y,z])
  - info['lidar2ego_translation']        ([x,y,z])
  - info['ego2global_rotation']          (quaternion [w,x,y,z])
  - info['ego2global_translation']       ([x,y,z])

This script keeps ALL legacy fields intact (so CustomNuScenesDataset still works)
and ADDS the new-format fields (so NuScenesMetric works without modification).
"""

import pickle
import numpy as np
from pyquaternion import Quaternion
from pathlib import Path
import sys


def quat_trans_to_4x4(rotation_quat, translation):
    """Build a 4x4 homogeneous transformation matrix from quaternion + translation."""
    q = Quaternion(rotation_quat)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = q.rotation_matrix
    mat[:3, 3] = np.array(translation, dtype=np.float64)
    return mat


def fix_entry(entry):
    """Add new-format calibration keys to a single legacy data_list entry."""
    # lidar_points.lidar2ego (4x4 matrix)
    if 'lidar_points' not in entry:
        entry['lidar_points'] = {}
    if 'lidar2ego' not in entry['lidar_points']:
        entry['lidar_points']['lidar2ego'] = quat_trans_to_4x4(
            entry['lidar2ego_rotation'],
            entry['lidar2ego_translation']
        )

    # ego2global (4x4 matrix)
    if 'ego2global' not in entry or not isinstance(entry.get('ego2global'), np.ndarray):
        entry['ego2global'] = quat_trans_to_4x4(
            entry['ego2global_rotation'],
            entry['ego2global_translation']
        )

    # images dict with cam2ego for each camera (needed by camera-path metrics)
    if 'images' not in entry and 'cams' in entry:
        images = {}
        for cam_type, cam_info in entry['cams'].items():
            cam_entry = {}
            if 'sensor2lidar_rotation' in cam_info and 'sensor2lidar_translation' in cam_info:
                # cam2ego = lidar2ego @ sensor2lidar
                sensor2lidar_r = np.array(cam_info['sensor2lidar_rotation'])
                sensor2lidar_t = np.array(cam_info['sensor2lidar_translation'])
                sensor2lidar = np.eye(4, dtype=np.float64)
                sensor2lidar[:3, :3] = sensor2lidar_r
                sensor2lidar[:3, 3] = sensor2lidar_t
                lidar2ego = entry['lidar_points']['lidar2ego']
                cam2ego = lidar2ego @ sensor2lidar
                cam_entry['cam2ego'] = cam2ego
            if 'cam_intrinsic' in cam_info:
                intrinsic = np.array(cam_info['cam_intrinsic'])
                viewpad = np.eye(4, dtype=np.float64)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                cam_entry['cam2img'] = viewpad
            if 'data_path' in cam_info:
                cam_entry['img_path'] = cam_info['data_path']
            images[cam_type] = cam_entry
        entry['images'] = images

    return entry


def fix_pkl(pkl_path):
    """Fix a single datalist PKL file in-place."""
    pkl_path = Path(pkl_path)
    print(f"\nProcessing: {pkl_path.name}")
    print(f"  Size: {pkl_path.stat().st_size / 1e6:.1f} MB")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or 'data_list' not in data:
        print(f"  SKIP: no 'data_list' key found")
        return

    data_list = data['data_list']
    print(f"  Entries: {len(data_list)}")

    # Check if already fixed
    sample = data_list[0]
    already_has_lidar_points = (
        'lidar_points' in sample
        and isinstance(sample['lidar_points'], dict)
        and 'lidar2ego' in sample['lidar_points']
    )
    already_has_ego2global_matrix = (
        'ego2global' in sample
        and isinstance(sample['ego2global'], np.ndarray)
    )
    if already_has_lidar_points and already_has_ego2global_matrix:
        print(f"  SKIP: already has new-format fields")
        return

    # Fix each entry
    for i, entry in enumerate(data_list):
        fix_entry(entry)

    # Verify fix
    fixed = data_list[0]
    assert 'lidar_points' in fixed and 'lidar2ego' in fixed['lidar_points'], \
        "lidar_points.lidar2ego missing after fix"
    assert isinstance(fixed['ego2global'], np.ndarray) and fixed['ego2global'].shape == (4, 4), \
        "ego2global not a 4x4 matrix after fix"
    assert 'token' in fixed, "token missing"
    # Verify legacy keys still present
    assert 'lidar_path' in fixed, "legacy lidar_path lost"
    assert 'cams' in fixed, "legacy cams lost"

    # Save
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"  FIXED: Added lidar_points.lidar2ego, ego2global (4x4), images.*.cam2ego")
    print(f"  New size: {pkl_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Verification: PASSED")


if __name__ == '__main__':
    nuscenes_root = Path(r'C:\datasets\nuscenes')

    pkls_to_fix = [
        nuscenes_root / 'nuscenes_infos_temporal_val_datalist.pkl',
        nuscenes_root / 'nuscenes_infos_temporal_val_boston_datalist.pkl',
        nuscenes_root / 'nuscenes_infos_temporal_val_singapore_datalist.pkl',
    ]

    print("=" * 60)
    print("Comprehensive PKL Fixer")
    print("Adds new-format calibration fields for NuScenesMetric")
    print("while preserving all legacy fields for CustomNuScenesDataset")
    print("=" * 60)

    for pkl in pkls_to_fix:
        if pkl.exists():
            fix_pkl(pkl)
        else:
            print(f"\n  NOT FOUND: {pkl.name}")

    print("\n" + "=" * 60)
    print("Done. All datalist PKLs now have both legacy and new-format fields.")
    print("=" * 60)
