# Dataset Status Report

**Generated:** 2026-03-13  
**Project:** Domain-Generalizable Camera-Only BEV 3D Detection  
**Data Root:** `C:\datasets\`

---

## Executive Summary

| Dataset | Raw Data | Extracted | Info Files (.pkl) | Ready for Training? |
|---------|----------|-----------|-------------------|---------------------|
| **nuScenes** | Partially complete | Yes | **Not generated** | No |
| **KITTI** | Complete | Yes | **Not generated** | No |

**Critical next step:** Run `create_data.py` for both nuScenes and KITTI to generate the `.pkl` info files required by BEVFormer.

---

## 1. nuScenes (Primary Training + In-Domain Evaluation)

### Location
`C:\datasets\nuscenes\`

### Current Status

| Component | Status | Details |
|-----------|--------|---------|
| **Metadata (v1.0-trainval)** | ✅ Complete | All JSON files present (scene, sample, sample_data, calibrated_sensor, ego_pose, etc.) |
| **samples/** (keyframe images) | ✅ Complete | **368,594 files** across 6 cameras |
| **sweeps/** (inter-frame images) | ⚠️ Partial | **171,034 camera sweep files missing** (14.4% of total) |
| **can_bus/** | ✅ Present | 7,832 files |
| **maps/** | ✅ Present | 4 map files |
| **nuscenes_infos_temporal_train.pkl** | ❌ Missing | Must run `create_data.py` |
| **nuscenes_infos_temporal_val.pkl** | ❌ Missing | Must run `create_data.py` |

### Missing Data Analysis (from `e:\bev_research\tools\investigation_results_final.txt`)

- **Total camera files expected:** 1,183,790  
- **Total camera files missing:** 171,034 (all in `sweeps/`)  
- **Scenes affected:** 170 scenes (mainly scene-0769 through scene-0883)  
- **Impact on BEVFormer:** **None** — BEVFormer uses only **keyframe cameras** (`samples/`) and **LIDAR sweeps** (not camera sweeps). The 171K missing files are camera sweeps, which BEVFormer does not load.

### Folder Structure (Current)

```
C:\datasets\nuscenes\
├── v1.0-trainval\          # Metadata JSON files
│   ├── scene.json
│   ├── sample.json
│   ├── sample_data.json
│   ├── calibrated_sensor.json
│   ├── ego_pose.json
│   └── ...
├── samples\                 # Keyframe images (complete)
│   ├── CAM_FRONT\
│   ├── CAM_FRONT_LEFT\
│   ├── CAM_FRONT_RIGHT\
│   ├── CAM_BACK\
│   ├── CAM_BACK_LEFT\
│   └── CAM_BACK_RIGHT\
├── sweeps\                  # Inter-frame images (partial)
├── can_bus\
├── maps\
└── (no .pkl files yet)
```

### What BEVFormer Needs

- **Keyframe images** (`samples/`) — ✅ **Complete**
- **LIDAR sweeps** (in `samples/` for LIDAR_TOP) — ✅ Present (verified via `get_available_scenes` LIDAR check)
- **Info files** — ❌ **Not generated**

### Required Action: Generate nuScenes Info Files

```powershell
cd E:\Auto_Image\BEVFormer
conda activate bev_research
python tools/create_data.py nuscenes `
    --root-path C:/datasets/nuscenes `
    --out-dir C:/datasets/nuscenes `
    --canbus C:/datasets/nuscenes `
    --extra-tag nuscenes `
    --version v1.0
```

This will create:
- `nuscenes_infos_temporal_train.pkl`
- `nuscenes_infos_temporal_val.pkl`
- `nuscenes_infos_temporal_test.pkl` (if v1.0-test exists)

---

## 2. KITTI 3D Object Detection (Cross-Domain Evaluation)

### Location
`C:\datasets\kitti\`

### Current Status

| Component | Status | Details |
|-----------|--------|---------|
| **training/image_2** | ✅ Complete | 7,481 images |
| **training/velodyne** | ✅ Complete | 7,481 point clouds |
| **training/calib** | ✅ Complete | 7,481 calibration files |
| **training/label_2** | ✅ Complete | 7,481 label files |
| **testing/image_2** | ✅ Complete | 7,518 images |
| **testing/velodyne** | ✅ Complete | 7,518 point clouds |
| **testing/calib** | ✅ Complete | Present |
| **ImageSets** | ✅ Present | train/val/test splits |
| **kitti_infos_train.pkl** | ❌ Missing | Must run `create_data.py` |
| **kitti_infos_val.pkl** | ❌ Missing | Must run `create_data.py` |
| **kitti_infos_test.pkl** | ❌ Missing | Must run `create_data.py` |

### Folder Structure (Current)

```
C:\datasets\kitti\
├── training\
│   ├── image_2\     # 7,481 images
│   ├── velodyne\    # 7,481 .bin files
│   ├── calib\       # 7,481 .txt files
│   └── label_2\     # 7,481 .txt files
├── testing\
│   ├── image_2\     # 7,518 images
│   ├── velodyne\    # 7,518 .bin files
│   └── calib\
├── ImageSets\
└── (no .pkl files yet)
```

### Required Action: Generate KITTI Info Files

```powershell
cd E:\Auto_Image\BEVFormer
conda activate bev_research
python tools/create_data.py kitti `
    --root-path C:/datasets/kitti `
    --out-dir C:/datasets/kitti `
    --extra-tag kitti
```

This will create:
- `kitti_infos_train.pkl`
- `kitti_infos_val.pkl`
- `kitti_infos_trainval.pkl`
- `kitti_infos_test.pkl`

**Note:** The BEVFormer `create_data.py` expects to be run from the `BEVFormer` directory (it has `sys.path.append('.')` and imports from `data_converter`). Ensure you're in the correct directory.

---

## 3. Other Data Locations

### `e:\raw data\`
- Contains ROS bag files and PNGs (~20 GB)
- **Not** nuScenes or KITTI format
- Not used by the current BEVFormer pipeline

### `e:\bev_research\data\` and `e:\Auto_Image\bev_research\data\`
- No datasets found here
- Configs use `C:\datasets\` (via `e:\bev_research` tools) or `./data/` (relative to BEVFormer)

---

## 4. Path Configuration Summary

| Codebase | nuScenes Path | KITTI Path |
|----------|---------------|------------|
| **e:\bev_research** | `C:/datasets/nuscenes/` | `C:/datasets/kitti` |
| **BEVFormer** (default) | `./data/nuscenes` | `./data/kitti` |
| **create_data.py** | Use `--root-path` and `--out-dir` | Same |

**Recommendation:** Either configure BEVFormer configs to use `C:/datasets/` directly, or create symlinks:
- `bev_research/data/nuscenes` → `C:\datasets\nuscenes`
- `bev_research/data/kitti` → `C:\datasets\kitti`

---

## 5. Info Files Expected by BEVFormer Configs

| Config Type | nuScenes | KITTI |
|-------------|----------|-------|
| BEVFormer (temporal) | `nuscenes_infos_temporal_train.pkl` | `kitti_infos_train.pkl` |
| BEVFormer (temporal) | `nuscenes_infos_temporal_val.pkl` | `kitti_infos_val.pkl` |
| nus-3d base | `nuscenes_infos_train.pkl` | — |

The `create_data.py` for nuScenes generates `*_temporal_*.pkl` files, which match the BEVFormer configs.

---

## 6. Checklist for Task 3 (Dataset Preparation)

- [ ] Run `create_data.py nuscenes` with `C:/datasets/nuscenes` paths
- [ ] Run `create_data.py kitti` with `C:/datasets/kitti` paths
- [ ] Verify `nuscenes_infos_temporal_{train,val}.pkl` exist
- [ ] Verify `kitti_infos_{train,val,test}.pkl` exist
- [ ] Update configs to use `C:/datasets/` (or ensure symlinks work)
- [ ] Update `notebooks/dataset_stats.md` with checkmarks
- [ ] (Optional) Add/run `tools/verify_data.py` for sanity checks

---

## 7. Optional: Download Missing nuScenes Camera Sweeps

If you want a **fully complete** nuScenes dataset (e.g., for other methods that use camera sweeps):

- **Missing:** 171,034 camera sweep files across 170 scenes
- **Scenes:** scene-0769 through scene-0883 (approximately)
- **Source:** nuScenes download page — re-download the sweeps archives for the affected scenes

For BEVFormer training and evaluation, **this is not required**.

---

*Last updated: 2026-03-13*
