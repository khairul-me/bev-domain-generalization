"""
One-shot converter: legacy BEVFormer PKL → mmdet3d NuScenesMetric format.

This script creates a SEPARATE metric-format PKL that contains exactly the
fields NuScenesMetric accesses. The dataset continues to read the original
legacy PKL unchanged.

Fields NuScenesMetric accesses (exhaustive grep of nuscenes_metric.py):
  Line 511: info['token']                               → string
  Line 652: info['lidar_points']['lidar2ego']            → 4×4 ndarray
  Line 663: info['ego2global']                           → 4×4 ndarray
  Line 698: info['images'][camera_type]['cam2ego']       → 4×4 ndarray
  Line 749: info['images']['CAM_FRONT']['cam2ego']       → 4×4 ndarray

No other fields are accessed. This script converts exactly these.
"""

import pickle
import numpy as np
from pyquaternion import Quaternion
from pathlib import Path


def quat_trans_to_4x4(rotation_quat, translation):
    """Build 4x4 homogeneous transform from quaternion + translation."""
    q = Quaternion(rotation_quat)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = q.rotation_matrix
    mat[:3, 3] = np.array(translation, dtype=np.float64)
    return mat


def convert_one_sample(legacy_info):
    """Convert a single legacy sample dict to metric-format dict."""
    new = {}

    # 1. token (string) — used at line 511
    new['token'] = legacy_info['token']

    # 2. lidar_points.lidar2ego (4×4) — used at line 652
    new['lidar_points'] = {
        'lidar2ego': quat_trans_to_4x4(
            legacy_info['lidar2ego_rotation'],
            legacy_info['lidar2ego_translation']
        ),
        'lidar_path': legacy_info.get('lidar_path', ''),
    }

    # 3. ego2global (4×4) — used at lines 663, 709, 737
    new['ego2global'] = quat_trans_to_4x4(
        legacy_info['ego2global_rotation'],
        legacy_info['ego2global_translation']
    )

    # 4. images[cam_type].cam2ego (4×4) — used at lines 698, 749
    if 'cams' in legacy_info:
        images = {}
        for cam_name, cam_info in legacy_info['cams'].items():
            cam2ego = quat_trans_to_4x4(
                cam_info['sensor2ego_rotation'],
                cam_info['sensor2ego_translation']
            )
            cam2img = np.eye(4, dtype=np.float64)
            intrinsic = np.array(cam_info['cam_intrinsic'], dtype=np.float64)
            cam2img[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

            images[cam_name] = {
                'cam2ego': cam2ego,
                'cam2img': cam2img,
                'img_path': cam_info.get('data_path', ''),
            }
        new['images'] = images

    # Extra fields that might be useful but aren't accessed by the metric
    new['timestamp'] = legacy_info.get('timestamp', 0)

    return new


def convert_pkl(legacy_path, output_path):
    """Convert a legacy PKL to metric-format PKL."""
    legacy_path = Path(legacy_path)
    output_path = Path(output_path)

    print(f"\nReading: {legacy_path.name} ({legacy_path.stat().st_size / 1e6:.1f} MB)")

    with open(legacy_path, 'rb') as f:
        legacy_data = pickle.load(f)

    # Handle both legacy wrapper formats
    if isinstance(legacy_data, dict) and 'infos' in legacy_data:
        infos = legacy_data['infos']
        metadata = legacy_data.get('metadata', {})
    elif isinstance(legacy_data, dict) and 'data_list' in legacy_data:
        infos = legacy_data['data_list']
        metadata = legacy_data.get('metainfo', {})
    else:
        raise ValueError(f"Unknown PKL format: keys={list(legacy_data.keys())}")

    print(f"  Samples: {len(infos)}")

    # CRITICAL: sort by timestamp to match CustomNuScenesDataset.load_data_list()
    # The dataset sorts: data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))
    # The metric PKL must be in the same order so sample_idx i maps to the same token.
    infos = list(sorted(infos, key=lambda e: e['timestamp']))
    print(f"  Sorted by timestamp: first_ts={infos[0]['timestamp']}, last_ts={infos[-1]['timestamp']}")

    # Check if legacy entries have the expected fields
    sample = infos[0]
    required_legacy = ['token', 'lidar2ego_rotation', 'lidar2ego_translation',
                       'ego2global_rotation', 'ego2global_translation']
    for key in required_legacy:
        if key not in sample:
            raise ValueError(f"  MISSING required legacy key: {key}")

    converted = [convert_one_sample(info) for info in infos]

    output = {
        'data_list': converted,
        'metainfo': metadata if isinstance(metadata, dict) else {'version': str(metadata)},
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"  Written: {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return converted


def validate(converted_samples):
    """Validate every sample has exactly the fields NuScenesMetric needs."""
    cam_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    for i, s in enumerate(converted_samples):
        # Field 1: token
        assert isinstance(s.get('token'), str), f"Sample {i}: token not a string"

        # Field 2: lidar_points.lidar2ego
        lp = s.get('lidar_points', {})
        l2e = lp.get('lidar2ego')
        assert isinstance(l2e, np.ndarray) and l2e.shape == (4, 4), \
            f"Sample {i}: lidar_points.lidar2ego not 4×4 ndarray"

        # Field 3: ego2global
        e2g = s.get('ego2global')
        assert isinstance(e2g, np.ndarray) and e2g.shape == (4, 4), \
            f"Sample {i}: ego2global not 4×4 ndarray"

        # Field 4: images[cam].cam2ego
        imgs = s.get('images', {})
        for cam in cam_types:
            assert cam in imgs, f"Sample {i}: missing camera {cam}"
            c2e = imgs[cam].get('cam2ego')
            assert isinstance(c2e, np.ndarray) and c2e.shape == (4, 4), \
                f"Sample {i}: images[{cam}].cam2ego not 4×4 ndarray"

    print(f"  Validation: ALL {len(converted_samples)} samples PASSED")


if __name__ == '__main__':
    nuscenes_root = Path(r'C:\datasets\nuscenes')

    conversions = [
        # (legacy_source, metric_output)
        (nuscenes_root / 'nuscenes_infos_temporal_val.pkl',
         nuscenes_root / 'nuscenes_infos_temporal_val_metric.pkl'),

        (nuscenes_root / 'nuscenes_infos_temporal_val_boston.pkl',
         nuscenes_root / 'nuscenes_infos_temporal_val_boston_metric.pkl'),

        (nuscenes_root / 'nuscenes_infos_temporal_val_singapore.pkl',
         nuscenes_root / 'nuscenes_infos_temporal_val_singapore_metric.pkl'),
    ]

    print("=" * 60)
    print("Legacy -> Metric Format PKL Converter")
    print("=" * 60)

    for legacy_path, metric_path in conversions:
        if not legacy_path.exists():
            print(f"\n  SKIP: {legacy_path.name} not found")
            continue
        converted = convert_pkl(legacy_path, metric_path)
        validate(converted)

    print("\n" + "=" * 60)
    print("Done. Use these metric PKLs for val_evaluator.ann_file")
    print("Keep the original legacy PKLs for dataset.ann_file")
    print("=" * 60)
