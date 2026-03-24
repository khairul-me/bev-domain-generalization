# FIX_GUIDE Implementation Status

**Date:** 2026-03-13

## Completed Fixes

### Fix 2 — mmcv Version Conflict
- **Status:** Resolved
- **Action:** Patched `E:\Auto_Image\BEVFormer\tools\data_converter\nuscenes_converter.py` — replaced `mmcv.track_iter_progress` with `track_iter_progress` (uses mmengine fallback)
- BEVFormer's create_data already had try/except for mmcv→mmengine; one remaining reference was fixed

### Fix 3 — Download KITTI ImageSets
- **Status:** Complete
- **Action:** Downloaded train.txt (3712), val.txt (3769), test.txt, trainval.txt to `C:\datasets\kitti\ImageSets\`
- **Verification:** All files present with correct line counts

### Fix 5 — Consolidate Project Structure
- **Status:** Complete
- **Action:** Moved TASK_PROGRESS_TRACKER, TASK1_VERIFICATION_REPORT, DATASET_STATUS_REPORT, COMPREHENSIVE_PROJECT_REVIEW to `E:\bev_research\docs\`
- **Action:** Copied literature_notes, depth_estimation_chain_bevformer, TASK2_READING_GUIDE to `E:\bev_research\notebooks\`

### Fix 6 — Replace Hardcoded Paths
- **Status:** Complete
- **Action:** Created `E:\bev_research\config\paths.py` with all path variables
- **Action:** Created `E:\bev_research\config\__init__.py`
- **Action:** Updated `depth_prior_module.py`, `run_kitti_prep.py`, `verify_data.py` to use config.paths

### Fix 7 — Correct pkl Filename References
- **Status:** Complete
- **Action:** Updated `verify_data.py` to use NUSCENES_TRAIN_PKL (nuscenes_infos_temporal_train.pkl), etc.
- **Action:** Config `bevformer_rtx5060.py` already had correct temporal filenames

---

## Pending Fixes (Require User Action)

### Fix 1 — Upgrade PyTorch
- **Status:** Blocked
- **Issue:** PyTorch 2.6+cu124 caused DLL load failure. Fallback to 2.5.1+cu121 failed with "Access denied" (file lock — close other Python/IDE processes)
- **Action required:**
  1. Close all Python processes, Cursor, and any IDE using the bev_research env
  2. Open a fresh PowerShell as Administrator
  3. Run:
     ```powershell
     conda activate bev_research
     pip uninstall torch torchvision -y
     pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
     ```
  4. Rebuild MMDet3D: `cd E:\Auto_Image\bev_research\mmdetection3d; pip install -v -e .`

### Fix 4 — Generate Dataset pkl Files
- **Status:** Blocked by Fix 1 (PyTorch)
- **Prerequisite:** PyTorch must import successfully
- **Command (after Fix 1):**
  ```powershell
  cd E:\Auto_Image\BEVFormer
  conda activate bev_research
  python tools/create_data.py nuscenes --root-path C:/datasets/nuscenes --out-dir C:/datasets/nuscenes --canbus C:/datasets/nuscenes --extra-tag nuscenes --version v1.0
  python tools/create_data.py kitti --root-path C:/datasets/kitti --out-dir C:/datasets/kitti --extra-tag kitti
  ```
- **Helper script:** `E:\bev_research\tools\run_create_data.py` (run with `python tools/run_create_data.py all`)

### Fix 8 — Download BEVFormer-Base Checkpoint
- **Status:** Not started
- **Action:** Download from BEVFormer GitHub, save to `E:\bev_research\checkpoints\bevformer_base_epoch_24.pth`

### Fix 9 — Run Task 4 Smoke Test
- **Status:** Blocked by Fix 1, 4, 8

---

## Quick Verification

After completing Fix 1 and Fix 4:

```powershell
cd E:\bev_research
conda activate bev_research
python -c "from config.paths import NUSCENES_ROOT, KITTI_ROOT; print('Paths OK')"
python tools/verify_data.py
```
