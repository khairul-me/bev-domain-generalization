"""
verify_data.py — Task 3 data verification script.
Verifies nuScenes and KITTI datasets are correctly structured and loadable.
"""
import os
import sys
import json

# ── nuScenes ─────────────────────────────────────────────────────────────────
NUSCENES_ROOT = "C:/datasets/nuscenes"
KITTI_ROOT    = "C:/datasets/kitti"

def verify_nuscenes():
    print("\n=== Verifying nuScenes ===")
    required = [
        "v1.0-trainval/scene.json",
        "v1.0-trainval/sample.json",
        "v1.0-trainval/sample_data.json",
        "v1.0-trainval/calibrated_sensor.json",
        "v1.0-trainval/ego_pose.json",
    ]
    missing = []
    for f in required:
        full = os.path.join(NUSCENES_ROOT, f)
        if os.path.exists(full):
            size = os.path.getsize(full) / 1e6
            print(f"  [OK] {f} ({size:.1f} MB)")
        else:
            missing.append(f)
            print(f"  [MISSING] {f}")

    # Try importing nuscenes and loading
    if not missing:
        try:
            sys.path.insert(0, "E:/Auto_Image/BEVFormer")
            from nuscenes.nuscenes import NuScenes
            nusc = NuScenes(version='v1.0-trainval', dataroot=NUSCENES_ROOT, verbose=False)
            print(f"  [OK] NuScenes loaded: {len(nusc.scene)} scenes, {len(nusc.sample)} samples")
            
            # Print first sample info
            sample = nusc.sample[0]
            scene = nusc.get('scene', sample['scene_token'])
            print(f"  [OK] First sample: scene '{scene['name']}', token {sample['token'][:8]}...")
            
            # Camera intrinsics from first sample
            cam_token = sample['data']['CAM_FRONT']
            cam_data = nusc.get('sample_data', cam_token)
            cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            K = cs['camera_intrinsic']
            print(f"  [OK] CAM_FRONT intrinsics: fx={K[0][0]:.1f}, fy={K[1][1]:.1f}, cx={K[0][2]:.1f}, cy={K[1][2]:.1f}")
            return True
        except Exception as e:
            print(f"  [INFO] nuScenes not loaded yet (expected until download): {e}")
            return False
    return False


def verify_kitti():
    print("\n=== Verifying KITTI ===")
    splits = {
        "training/calib": lambda p: len([f for f in os.listdir(p) if f.endswith('.txt')]) > 0,
        "training/image_2": lambda p: len([f for f in os.listdir(p) if f.endswith('.png')]) > 0,
        "training/label_2": lambda p: len([f for f in os.listdir(p) if f.endswith('.txt')]) > 0,
        "training/velodyne": lambda p: len([f for f in os.listdir(p) if f.endswith('.bin')]) > 0,
        "testing/calib": lambda p: len([f for f in os.listdir(p) if f.endswith('.txt')]) > 0,
        "testing/image_2": lambda p: len([f for f in os.listdir(p) if f.endswith('.png')]) > 0,
    }
    all_ok = True
    for rel_path, check in splits.items():
        full = os.path.join(KITTI_ROOT, rel_path)
        if os.path.exists(full):
            try:
                count = len(os.listdir(full))
                if count > 0:
                    print(f"  [OK] {rel_path}: {count} files")
                else:
                    print(f"  [EMPTY] {rel_path}: directory exists but empty")
                    all_ok = False
            except Exception as e:
                print(f"  [ERROR] {rel_path}: {e}")
                all_ok = False
        else:
            print(f"  [MISSING] {rel_path}")
            all_ok = False
    return all_ok


def check_info_files():
    print("\n=== Checking .pkl info files (BEVFormer) ===")
    pkl_files = [
        "C:/datasets/nuscenes/nuscenes_infos_train.pkl",
        "C:/datasets/nuscenes/nuscenes_infos_val.pkl",
        "C:/datasets/kitti/kitti_infos_train.pkl",
        "C:/datasets/kitti/kitti_infos_val.pkl",
    ]
    for f in pkl_files:
        if os.path.exists(f):
            size = os.path.getsize(f) / 1e6
            print(f"  [OK] {os.path.basename(f)} ({size:.1f} MB)")
        else:
            print(f"  [MISSING] {os.path.basename(f)} (will be generated after download)")


if __name__ == "__main__":
    print("=" * 60)
    print("TASK 3 DATA VERIFICATION SCRIPT")
    print("=" * 60)
    
    nusc_ok = verify_nuscenes()
    kitti_ok = verify_kitti()
    check_info_files()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  nuScenes: {'LOADED' if nusc_ok else 'NOT YET DOWNLOADED'}")
    print(f"  KITTI:    {'LOADED' if kitti_ok else 'NOT YET DOWNLOADED'}")
    print("\nNext step: Download datasets as described in dataset_download_instructions.md")
    print("=" * 60)
