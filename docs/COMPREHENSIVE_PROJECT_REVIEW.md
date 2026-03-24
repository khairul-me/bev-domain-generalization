# Comprehensive Project Review

**Date:** 2026-03-13  
**Project:** Domain-Generalizable Camera-Only BEV 3D Detection via Foundation Depth Priors  
**Reference:** `e:\Auto_Image\research_pipeline.md` (15-task plan)

---

## 1. Executive Summary

| Aspect | Status | Summary |
|--------|--------|---------|
| **Environment** | ✅ Complete | Conda `bev_research`, PyTorch 2.1+cu121, MMDet3D, BEVFormer, Depth Anything V2 |
| **Literature** | 🟡 In Progress | Notes in `notebooks/literature_notes.md`, depth chain diagram created |
| **Datasets** | ⚠️ Blocked | Raw data present; `.pkl` info files not generated; KITTI missing ImageSets |
| **Codebase** | ⚠️ Fragmented | Two project roots, multiple BEVFormer copies, path inconsistencies |
| **Disk Space** | ✅ Adequate | C: 170 GB free, E: 309 GB free; experiments should use E: |
| **Next Steps** | — | Fix KITTI ImageSets → run create_data.py → consolidate project structure |

---

## 2. Directory Structure (Complete)

### 2.1 Primary Locations

```
E:\Auto_Image\                          # Main workspace (research_pipeline)
├── research_pipeline.md                # Master 15-task plan
├── BEVFormer\                          # Standalone BEVFormer (fundamentalvision)
│   ├── projects\                       # Configs, mmdet3d_plugin
│   ├── tools\                          # create_data.py, train.py, test.py
│   └── tools\data_converter\           # nuscenes_converter, kitti_converter
├── bev_research\                       # Project from pipeline (Tasks 1–2)
│   ├── mmdetection3d\                  # MMDet3D with BEVFormer projects
│   ├── BEVFormer\                      # Duplicate copy
│   ├── Depth-Anything-V2\              # Symlink or copy
│   ├── notebooks\                      # literature_notes, depth_estimation_chain
│   ├── tools\                          # verify_environment.py
│   ├── TASK_PROGRESS_TRACKER.md        # Task status
│   ├── TASK1_VERIFICATION_REPORT.md
│   ├── DATASET_STATUS_REPORT.md
│   └── COMPREHENSIVE_PROJECT_REVIEW.md  # This file
├── Depth-Anything-V2\                  # Depth Anything V2 repo
└── tools\verify_env.py

E:\bev_research\                        # Separate implementation (more advanced)
├── checkpoints\                        # bevformer_tiny_fp16_epoch_24.pth, depth_anything_v2_vits.pth
├── configs\                            # bevformer_rtx5060.py, bevformer_depth_prior_finetune.py
├── data\                               # kitti/, nuscenes/ (empty or symlinks)
├── experiments\                        # baseline/, with_adapter/, with_tta/
│   ├── baseline\                       # results.md, failure_analysis.md
│   └── ...
├── models\                             # depth_adapter/, tta_module/ (actual code)
├── notebooks\                          # dataset_stats.md, literature_notes.md
├── tools\                              # investigate_missing_data.py, run_kitti_prep.py, verify_data.py
└── paper\outline.md

C:\datasets\                            # Dataset root (369 GB total)
├── nuscenes\                           # ~350 GB
│   ├── v1.0-trainval\                  # Metadata JSON
│   ├── samples\                        # 368,594 keyframe images ✅
│   ├── sweeps\                         # Partial (171K camera sweeps missing)
│   ├── can_bus\                        # 7,832 files
│   └── maps\
└── kitti\                              # ~41 GB
    ├── training\                       # image_2, velodyne, calib, label_2 (7,481 each) ✅
    ├── testing\                        # image_2, velodyne, calib (7,518 each) ✅
    └── ImageSets\                      # ⚠️ EMPTY — train.txt, val.txt, etc. MISSING

E:\raw data\                            # 21 GB — ROS bags, PNGs (not nuScenes/KITTI)
```

### 2.2 Key File Inventory

| File | Location | Purpose |
|------|----------|---------|
| `research_pipeline.md` | E:\Auto_Image\ | 15-task master plan |
| `TASK_PROGRESS_TRACKER.md` | E:\Auto_Image\bev_research\ | Task 1–5 status |
| `DATASET_STATUS_REPORT.md` | E:\Auto_Image\bev_research\ | Dataset status |
| `create_data.py` | E:\Auto_Image\BEVFormer\tools\ | BEVFormer data prep (nuscenes, kitti) |
| `create_data.py` | E:\Auto_Image\bev_research\mmdetection3d\tools\ | MMDet3D data prep (different API) |
| `bevformer_rtx5060.py` | E:\bev_research\configs\ | RTX 5060 config, uses C:/datasets/nuscenes |
| `run_kitti_prep.py` | E:\bev_research\tools\ | Wrapper for KITTI create_data (uses E:/Auto_Image/BEVFormer) |
| `investigation_results_final.txt` | E:\bev_research\tools\ | nuScenes missing-file analysis |
| `dataset_stats.md` | E:\bev_research\notebooks\ | Dataset stats (updated) |

---

## 3. Task Status (Pipeline Alignment)

### Task 1: Environment Setup — ✅ COMPLETE

| Checkpoint | Status | Evidence |
|------------|--------|----------|
| nvidia-smi shows RTX 5060 | ✅ | TASK1_VERIFICATION_REPORT.md |
| torch.cuda.is_available() | ✅ | — |
| MMDetection3D | ✅ | mmdet3d 1.4.0 |
| BEVFormer cloned | ✅ | E:\Auto_Image\BEVFormer, projects in mmdetection3d |
| Depth Anything V2 | ✅ | E:\Auto_Image\Depth-Anything-V2 |
| Conda env bev_research | ✅ | Python 3.9 |

**Note:** RTX 5060 Ti (sm_120) has PyTorch compatibility warning; CUDA still works.

---

### Task 2: Focused Literature Review — 🟡 IN PROGRESS

| Deliverable | Status | Location |
|-------------|--------|----------|
| Literature notes (10 papers) | ✅ | notebooks/literature_notes.md |
| Depth estimation chain diagram | ✅ | notebooks/depth_estimation_chain_bevformer.md |
| TASK2_READING_GUIDE.md | ✅ | notebooks/ |
| Verification checklist | ✅ | TASK2_VERIFICATION_CHECKLIST.md |

---

### Task 3: Dataset Acquisition & Preparation — ⚠️ BLOCKED

| Step | Status | Blocker / Notes |
|------|--------|-----------------|
| nuScenes download | ✅ | 368,594 samples; 171K camera sweeps missing (not used by BEVFormer) |
| nuScenes create_data.py | ❌ | Not run; pkl files missing |
| KITTI download | ✅ | training + testing complete |
| KITTI ImageSets | ❌ | **train.txt, val.txt, test.txt, trainval.txt MISSING** |
| KITTI create_data.py | ❌ | Failed (see prep_log*.txt) |
| dataset_stats.md | ✅ | Updated in E:\bev_research\notebooks |
| verify_data.py | ✅ | Exists in E:\bev_research\tools |

**KITTI ImageSets:** The official KITTI download does NOT include ImageSets. They must be downloaded separately from [second.pytorch](https://github.com/traveller59/second.pytorch) or MMDet3D docs:
```
wget https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt -O C:\datasets\kitti\ImageSets\train.txt
wget https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt -O C:\datasets\kitti\ImageSets\val.txt
wget https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt -O C:\datasets\kitti\ImageSets\test.txt
wget https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt -O C:\datasets\kitti\ImageSets\trainval.txt
```

**create_data.py failures (from C:\datasets\kitti\prep_log*.txt):**
1. **prep_log.txt:** `ImportError: cannot import name 'track_iter_progress' from 'mmcv'` — mmcv/mmengine version mismatch
2. **prep_log_retry4.txt:** `FileNotFoundError: C:\datasets\kitti\ImageSets\train.txt` — ImageSets not populated

---

### Task 4: Baseline Codebase Setup — ⬜ NOT STARTED

Blocked by Task 3 (no pkl files). E:\bev_research has configs and checkpoints prepared but cannot run without data.

---

### Task 5: Baseline Reproduction — ⬜ NOT STARTED

Blocked by Task 3. `experiments/baseline/results.md` exists with placeholder tables (TBD).

---

### Tasks 6–15 — ⬜ NOT STARTED

E:\bev_research has scaffolding: `models/depth_adapter/`, `models/tta_module/`, `experiments/with_adapter/`, `experiments/with_tta/`.

---

## 4. Disk Space Analysis

### Current Usage

| Drive | Used | Free | Total | Notes |
|-------|------|------|-------|-------|
| **C:** | 690.75 GB | **170.6 GB** | 861.36 GB | Datasets (~369 GB) on C:\datasets |
| **E:** | 191.15 GB | **308.85 GB** | 500 GB | Code, checkpoints, experiments |

### Training Requirements (Colab-like)

| Item | Estimate | Notes |
|------|----------|-------|
| nuScenes data | ~350 GB | Already on C: |
| KITTI data | ~41 GB | Already on C: |
| pkl files | ~2.5 GB | To be generated on C: |
| Model checkpoints | ~2–5 GB | BEVFormer ~382 MB, DAv2 ~400 MB |
| Experiment outputs | 20–50 GB | Logs, ckpts, visualizations per run |
| Temp / cache | 5–10 GB | During training |

**Recommendation:**
- **C: drive:** Keep datasets; 170 GB free is sufficient for pkl generation and temp files.
- **E: drive:** Use for `--work-dir` (experiments), checkpoints, and logs. 309 GB free is ample for multiple training runs.
- **Colab comparison:** Colab typically has ~78 GB disk. Your E: drive has 4× that free — more than adequate.

---

## 5. Inconsistencies & Issues

### 5.1 Two Project Roots

| Location | Role | Used By |
|----------|------|---------|
| **E:\Auto_Image\bev_research** | Pipeline Tasks 1–2, TASK_PROGRESS_TRACKER | research_pipeline.md |
| **E:\bev_research** | Implementation, configs, models, experiments | run_kitti_prep, bevformer_rtx5060, test scripts |

**Issue:** Unclear which is the canonical project. Configs in E:\bev_research reference E:/Auto_Image/BEVFormer. TASK_PROGRESS_TRACKER lives in E:\Auto_Image\bev_research.

**Recommendation:** Choose one canonical root. If E:\bev_research is the active implementation, migrate TASK_PROGRESS_TRACKER and ensure all paths are consistent.

---

### 5.2 Multiple BEVFormer Copies

| Copy | Use Case |
|------|----------|
| E:\Auto_Image\BEVFormer | Standalone; create_data.py, train.py, test.py |
| E:\Auto_Image\bev_research\mmdetection3d | MMDet3D with projects; different create_data API |
| E:\Auto_Image\bev_research\BEVFormer | Duplicate of standalone |

E:\bev_research configs use `_base_ = ['E:/Auto_Image/BEVFormer/projects/configs/...']` — they depend on the standalone BEVFormer, not mmdetection3d.

---

### 5.3 create_data.py Source Confusion

- **BEVFormer** (`E:\Auto_Image\BEVFormer\tools\create_data.py`): Uses `data_converter` (mmcv), generates `nuscenes_infos_temporal_*.pkl` for nuScenes.
- **MMDet3D** (`bev_research\mmdetection3d\tools\create_data.py`): Uses `dataset_converters`, different API; for KITTI can create ImageSets via `create_ImageSets_img_ids` (Waymo flow).

**run_kitti_prep.py** uses BEVFormer's create_data. BEVFormer expects ImageSets to already exist.

---

### 5.4 pkl Naming Mismatch

| Source | nuScenes pkl | KITTI pkl |
|--------|--------------|-----------|
| **Pipeline (Task 3)** | nuscenes_infos_train.pkl | kitti_infos_train.pkl |
| **BEVFormer create_data** | nuscenes_infos_**temporal**_train.pkl | kitti_infos_train.pkl |
| **BEVFormer configs** | nuscenes_infos_temporal_train.pkl | kitti_infos_train.pkl |
| **verify_data.py** | nuscenes_infos_train.pkl | kitti_infos_train.pkl |

**Fix:** verify_data.py should check for `nuscenes_infos_temporal_train.pkl` to match BEVFormer.

---

### 5.5 Path Hardcoding

Several files hardcode paths:
- `E:\bev_research\models\depth_adapter\depth_prior_module.py`: `sys.path.insert(0, 'E:/Auto_Image/Depth-Anything-V2')`
- `E:\bev_research\configs\bevformer_rtx5060.py`: `_base_ = ['E:/Auto_Image/BEVFormer/...']`
- `E:\bev_research\tools\run_kitti_prep.py`: `os.chdir('E:/Auto_Image/BEVFormer')`
- `E:\bev_research\tools\verify_data.py`: `sys.path.insert(0, "E:/Auto_Image/BEVFormer")`

These assume E:\Auto_Image exists. If project moves, all break.

---

## 6. What Has Been Done vs. What Remains

### Completed

1. **Environment:** Conda env, PyTorch, MMDet3D, BEVFormer, Depth Anything V2
2. **Literature:** Notes for 10 papers, depth chain diagram
3. **Datasets:** nuScenes and KITTI raw data downloaded and extracted
4. **Scaffolding:** E:\bev_research has configs, models (depth_adapter, tta_module), experiments layout, checkpoints (BEVFormer-Tiny fp16, DAv2 ViT-S)
5. **Investigation:** nuScenes missing-file analysis (171K camera sweeps; BEVFormer unaffected)
6. **Documentation:** TASK_PROGRESS_TRACKER, DATASET_STATUS_REPORT, dataset_stats, verification reports

### Remaining (in order)

1. **Download KITTI ImageSets** (train.txt, val.txt, test.txt, trainval.txt) into `C:\datasets\kitti\ImageSets\`
2. **Fix mmcv/mmengine** if create_data fails on `track_iter_progress` (BEVFormer may need mmcv 1.x)
3. **Run create_data.py for nuScenes** (from E:\Auto_Image\BEVFormer)
4. **Run create_data.py for KITTI** (from E:\Auto_Image\BEVFormer)
5. **Verify data loading** with verify_data.py (and fix pkl names)
6. **Consolidate project structure** — decide canonical root, update paths
7. **Task 4–5:** Smoke test, baseline reproduction on nuScenes val
8. **Tasks 6–15:** Per pipeline

---

## 7. Recommended Immediate Actions

### Priority 1: Unblock Dataset Preparation

```powershell
# 1. Download KITTI ImageSets (run from PowerShell)
$base = "https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets"
$out = "C:\datasets\kitti\ImageSets"
New-Item -ItemType Directory -Force -Path $out
Invoke-WebRequest "$base/train.txt" -OutFile "$out/train.txt"
Invoke-WebRequest "$base/val.txt" -OutFile "$out/val.txt"
Invoke-WebRequest "$base/test.txt" -OutFile "$out/test.txt"
Invoke-WebRequest "$base/trainval.txt" -OutFile "$out/trainval.txt"

# 2. Run nuScenes create_data
cd E:\Auto_Image\BEVFormer
conda activate bev_research
python tools/create_data.py nuscenes --root-path C:/datasets/nuscenes --out-dir C:/datasets/nuscenes --canbus C:/datasets/nuscenes --extra-tag nuscenes --version v1.0

# 3. Run KITTI create_data (after ImageSets exist)
python tools/create_data.py kitti --root-path C:/datasets/kitti --out-dir C:/datasets/kitti --extra-tag kitti
```

### Priority 2: Resolve mmcv Import (if needed)

If `track_iter_progress` fails, BEVFormer's create_data expects mmcv 1.x. Check:
```python
# In bev_research env
python -c "from mmcv import track_iter_progress; print('OK')"
```
If it fails, consider using mmengine's `track_iter_progress` or installing mmcv 1.x alongside mmdet3d.

### Priority 3: Consolidate Documentation

- Move or symlink TASK_PROGRESS_TRACKER to E:\bev_research if that is the active project
- Update verify_data.py to check for `nuscenes_infos_temporal_*.pkl`
- Add a single PROJECT_ROOT or config that all scripts reference

---

## 8. Checklist Before Training

- [ ] KITTI ImageSets/train.txt, val.txt, test.txt, trainval.txt exist
- [ ] nuscenes_infos_temporal_train.pkl and nuscenes_infos_temporal_val.pkl exist
- [ ] kitti_infos_train.pkl and kitti_infos_val.pkl exist
- [ ] verify_data.py runs without errors
- [ ] Config data_root points to C:/datasets/nuscenes (or correct path)
- [ ] work-dir for training points to E:\bev_research\experiments\ (or E: drive)
- [ ] Conda env bev_research activates and imports succeed

---

*Report generated 2026-03-13. Update after completing Task 3.*
