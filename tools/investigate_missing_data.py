
import os
import json
from nuscenes.nuscenes import NuScenes

dataroot = 'C:/datasets/nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

missing_files = []
camera_sensors = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

for sd in nusc.sample_data:
    if sd['sensor_modality'] == 'camera':
        path = os.path.join(dataroot, sd['filename'])
        if not os.path.exists(path):
            missing_files.append(sd['filename'])

print(f"\nTotal camera files expected: {len([s for s in nusc.sample_data if s['sensor_modality'] == 'camera'])}")
print(f"Total camera files missing: {len(missing_files)}")

if missing_files:
    print("\nFirst 10 missing files:")
    for f in missing_files[:10]:
        print(f"  {f}")
    
    # Try to group by scene
    missing_by_scene = {}
    for f in missing_files:
        # File pattern usually contains scene info or we can lookup via metadata
        # But let's just group by the part of the path that might indicate the blob
        pass

    # Better: use metadata to group missing files by scene
    scene_missing = {}
    for sd in nusc.sample_data:
        if sd['sensor_modality'] == 'camera':
            path = os.path.join(dataroot, sd['filename'])
            if not os.path.exists(path):
                sample = nusc.get('sample', sd['sample_token'])
                scene = nusc.get('scene', sample['scene_token'])
                scene_name = scene['name']
                scene_missing[scene_name] = scene_missing.get(scene_name, 0) + 1
    
    print(f"\nTotal scenes with missing files: {len(scene_missing)}")
    sorted_scenes = sorted(scene_missing.items(), key=lambda x: x[1], reverse=True)
    print("\nScenes with most missing camera samples:")
    for name, count in sorted_scenes[:20]:
        print(f"  {name}: {count} samples missing")
