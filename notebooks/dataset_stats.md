# Dataset Statistics & Documentation

**Task 3** — Populated 2026-03-11  
**Dataset Root:** `C:\datasets\`

---

## 1. nuScenes (Primary Training + In-Domain Evaluation)

### Download Status
- [ ] v1.0-trainval metadata downloaded
- [ ] Sensor data (sweeps + samples) downloaded (~300 GB)
- [ ] nuscenes_infos_train.pkl generated
- [ ] nuscenes_infos_val.pkl generated

### Dataset Properties
| Property | Value |
|----------|-------|
| Location | Boston, MA + Singapore |
| Sensor rig | 6 cameras (360°), 32-beam LiDAR, radar |
| scenes | 1000 total: 700 train, 150 val, 150 test |
| Labeled samples | ~40,000 keyframe samples |
| Image resolution | 1600 × 900 |
| **Focal length (fx)** | **~1266 px** |
| **Focal length (fy)** | **~1266 px** |
| **Principal point (cx)** | **~800 px** |
| **Principal point (cy)** | **~450 px** |
| Ego-vehicle | SUV-class, camera height ~1.8 m |
| Detection classes | 10 (car, truck, bus, trailer, construction_vehicle, pedestrian, motorcycle, bicycle, traffic_cone, barrier) |
| Annotation density | ~1400 3D boxes/scene |

### NDS Metric
```
NDS = (1/10) × [5×mAP + Σ(1 - min(1, mTPₑ))]
```
- mAP: center-distance matching at 0.5, 1, 2, 4 m thresholds
- TP errors: mATE (m), mASE (1 - 3D_IoU), mAOE (rad), mAVE (m/s), mAAE (1 - acc)

### Download Instructions
1. Register at https://www.nuscenes.org/nuscenes
2. Download `v1.0-trainval` metadata files (JSON, ~300 MB)
3. Download all `samples` and `sweeps` archives (~300 GB total)
4. Extract maintaining original structure to `C:\datasets\nuscenes\`

```
C:\datasets\nuscenes\
├── v1.0-trainval\
│   ├── scene.json
│   ├── sample.json
│   ├── sample_data.json
│   ├── calibrated_sensor.json
│   └── ego_pose.json
├── samples\
│   ├── CAM_FRONT\
│   ├── CAM_BACK\
│   ├── CAM_FRONT_LEFT\
│   ├── CAM_FRONT_RIGHT\
│   ├── CAM_BACK_LEFT\
│   └── CAM_BACK_RIGHT\
└── sweeps\
```

### Generate Info Files (run after download)
```bash
cd E:\Auto_Image\BEVFormer
conda activate bev_research
python tools/create_data.py nuscenes \
    --root-path C:/datasets/nuscenes \
    --out-dir C:/datasets/nuscenes \
    --extra-tag nuscenes
```

---

## 2. KITTI 3D Object Detection (Cross-Domain Evaluation)

### Download Status
- [ ] Left color images downloaded (12 GB)
- [ ] Velodyne point clouds downloaded (29 GB)
- [ ] Camera calibration downloaded
- [ ] Training labels downloaded
- [ ] kitti_infos_train.pkl generated
- [ ] kitti_infos_val.pkl generated

### Dataset Properties
| Property | Value |
|----------|-------|
| Location | Karlsruhe, Germany (suburban roads) |
| Sensors | Stereo cameras, 64-beam Velodyne LiDAR |
| Labeled frames | 7481 training, 7518 testing |
| Image resolution | 1242 × 375 |
| **Focal length (fx)** | **~721 px** |
| **Focal length (fy)** | **~721 px** |
| **Principal point (cx)** | **~609 px** |
| **Principal point (cy)** | **~172 px** |
| Ego-vehicle | Sedan-class, camera height ~1.4 m |
| Detection classes | 3 (Car, Pedestrian, Cyclist) |
| 3D AP IoU threshold | 0.7 (Car), 0.5 (Ped/Cyc) |
| Difficulty levels | Easy / Moderate / Hard |

### Download Instructions
1. Register at https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
2. Download:
   - Left color images of object data set (12 GB)
   - Velodyne point clouds (29 GB)
   - Camera calibration matrices
   - Training labels of object data set
3. Extract to `C:\datasets\kitti\`

### Generate Info Files (run after download)
```bash
cd E:\Auto_Image\BEVFormer
python tools/create_data.py kitti \
    --root-path C:/datasets/kitti \
    --out-dir C:/datasets/kitti \
    --extra-tag kitti
```

---

## 3. Domain Gap Analysis

### Camera Intrinsics Comparison
| Property | nuScenes | KITTI | Ratio / Impact |
|----------|----------|-------|----------------|
| Focal length fx | 1266 px | 721 px | **1.75× difference** |
| Image width | 1600 px | 1242 px | 1.29× |
| Image height | 900 px | 375 px | 2.4× |
| Camera height | ~1.8 m | ~1.4 m | 0.22 m offset |
| Horizontal FoV | ~60° | ~80° | Wider in KITTI |

### Depth Distribution Shift (Predicted)
- A car at 20m depth that projects to image row Y_ns in nuScenes will project to image row Y_k ≈ 1.75 × Y_ns in KITTI
- BEVFormer trains reference points based on nuScenes geometry → at inference on KITTI, these reference points sample from wrong image locations
- The depth scale learned by the network is embedded in the attention weights and is domain-specific

### Class Distribution (Approximate)
| Class | nuScenes Train | KITTI Train |
|-------|---------------|-------------|
| Car | ~600K boxes | ~28,742 |
| Pedestrian | ~240K boxes | ~4,487 |
| Bicycle/Cyclist | ~55K boxes | ~1,627 |
| Truck/Bus/Trailer | ~200K boxes | N/A |

---

## 4. Disk Space Budget

| Dataset | Estimated Size | Drive | Notes |
|---------|---------------|-------|-------|
| nuScenes trainval | ~300 GB | C:\ | Main download |
| KITTI | ~41 GB | C:\ | Training + testing |
| nuScenes pkl files | ~2 GB | C:\ | Generated |
| KITTI pkl files | ~0.5 GB | C:\ | Generated |
| Model checkpoints | ~20 GB | E:\ | BEVFormer + DAv2 |
| Experiment outputs | ~50 GB | E:\ | Logs, figures |
| **Total** | **~413 GB** | | C: has 478 GB free ✓ |

---

*Last updated: 2026-03-11*
