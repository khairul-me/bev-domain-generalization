from nuscenes.nuscenes import NuScenes
import os

nusc = NuScenes(version='v1.0-trainval', dataroot='C:/datasets/nuscenes', verbose=False)
missing_cameras = []
for sample in nusc.sample:
    for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        sd = nusc.get('sample_data', sample['data'][cam])
        path = os.path.join('C:/datasets/nuscenes', sd['filename'])
        if not os.path.exists(path):
            missing_cameras.append(path)

print(f"Missing camera files: {len(missing_cameras)}")
