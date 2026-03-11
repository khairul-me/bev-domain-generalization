# experiments/baseline/results.md
# Task 5: Baseline Reproduction Results

## BEVFormer-Tiny fp16 Baseline Results — nuScenes val

**Checkpoint:** `checkpoints/bevformer_tiny_fp16_epoch_24.pth` (382 MB)  
**Config:** `configs/bevformer_rtx5060.py` (inherits BEVFormer-Tiny fp16)  
**Evaluation date:** TBD (requires nuScenes download)

### Published vs Reproduced (nuScenes val)

| Metric | Published | Reproduced | Delta |
|--------|-----------|------------|-------|
| mAP    | 25.7%     | TBD        | –     |
| NDS    | 35.9%     | TBD        | –     |
| mATE   | 0.978     | TBD        | –     |
| mASE   | 0.277     | TBD        | –     |
| mAOE   | 0.428     | TBD        | –     |
| mAVE   | 0.861     | TBD        | –     |
| mAAE   | 0.199     | TBD        | –     |

*Note: Numbers from BEVFormer official GitHub, 24-epoch BEVFormer-Tiny-fp16.*

### Evaluation Command (run after nuScenes download)
```bash
cd E:\Auto_Image\BEVFormer
conda activate bev_research
python tools/test.py \
    projects/configs/bevformer_fp16/bevformer_tiny_fp16.py \
    E:/bev_research/checkpoints/bevformer_tiny_fp16_epoch_24.pth \
    --eval bbox \
    --out E:/bev_research/experiments/baseline/results_nuscenes_val.pkl
```

---

## Cross-Domain Failure: nuScenes-Trained BEVFormer on KITTI

**Goal:** Document zero-shot transfer failure as core paper motivation.

### Expected Failure Modes
1. **Depth scale collapse:** Objects detected at wrong depth range (scaled by fx ratio 1266/721 ≈ 1.75×)
2. **Misaligned 3D boxes:** Reference points project to wrong image regions
3. **Low detection confidence:** Features sampled from wrong locations → poor matching
4. **Phantom detections:** nuScenes class labels may not map to KITTI classes

### Cross-Domain Evaluation Command (run after KITTI download)
```bash
cd E:\Auto_Image\BEVFormer
python E:\bev_research\tools\test_cross_domain.py \
    --checkpoint E:/bev_research/checkpoints/bevformer_tiny_fp16_epoch_24.pth \
    --dataset kitti \
    --out E:/bev_research/experiments/baseline/results_kitti_cross_domain.pkl
```

### Cross-Domain Results Table

| Method | nuScenes NDS (in-domain) | KITTI Car AP/Easy (cross-domain) | Generalization Drop |
|--------|--------------------------|----------------------------------|---------------------|
| BEVFormer-Tiny-fp16 | 35.9% (pub) / TBD (repr) | TBD | TBD |
| Ours (with DAv2 + TTA) | TBD | TBD | TBD |

---

*Created: 2026-03-11 | Update after dataset download and evaluation*
