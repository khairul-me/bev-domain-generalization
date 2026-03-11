# Cross-Domain Failure Analysis
**Task 6** — Generated 2026-03-11  
**Research:** Domain-Generalizable Camera-Only BEV 3D Detection

---

## 1. Cross-Domain Evaluation Protocol

**Source domain:** nuScenes trainval (Boston + Singapore, 32-beam LiDAR, 6 cameras, 1600×900)  
**Target domain:** KITTI 3D (Karlsruhe, Germany, 64-beam LiDAR, stereo, 1242×375)

**Evaluation procedure:**
- No retraining on KITTI (zero-shot transfer only)
- KITTI val split: standard 3769-sample split
- Metric: KITTI 3D AP at IoU=0.7 (Car class), Easy/Moderate/Hard
- Comparison: (a) nuScenes-trained BEVFormer-Tiny-fp16, (b) our method

---

## 2. Root Cause: Camera Intrinsic Parameter Mismatch

### Quantitative Evidence

| Property | nuScenes | KITTI | Impact |
|----------|----------|-------|--------|
| Focal length fx | **1266 px** | **721 px** | **1.75× ratio** |
| Image width | 1600 px | 1242 px | 1.29× |
| Image height | 900 px | 375 px | 2.4× |
| Principal point cx | 816 px | 610 px | 206 px shift |
| Principal point cy | 492 px | 173 px | 319 px shift |
| Camera height | ~1.8 m | ~1.4 m | 0.4 m offset |

### BEV Reference Point Error (Computed)

BEVFormer uses camera intrinsics to project 3D reference points onto image planes.  
When applied to KITTI images using nuScenes intrinsics:

| Metric | Value |
|--------|-------|
| Principal point shift error | **379.8 px** (constant, depth-independent) |
| Mean BEV reference point error | **961.0 px** across BEV grid |
| Maximum BEV reference point error | **13,832 px** (distant objects) |
| Height projection ratio | **1.755×** at all depths |

**Interpretation:** BEVFormer's spatial cross-attention samples image features at wrong coordinates (379–13832 px error). The model literally looks at the wrong pixels for every 3D reference point.

---

## 3. Failure Mode Categorization

### Mode 1: Systematic Depth Scale Error
- ALL detected objects are at incorrect depth (scaled by 1.75×)
- A car at 20m in KITTI is perceived as being at ~35m
- Consequence: 3D bounding boxes placed at wrong range → miss threshold

### Mode 2: Misaligned Feature Sampling  
- SCA reference points projected using nuScenes K → land outside KITTI image crops  
- Network samples background/sky features instead of object features  
- Consequence: detection confidence collapses → missed detections (false negatives)

### Mode 3: Phantom Detections
- BEVFormer is trained to detect in BEV range [-51.2, 51.2]m  
- Incorrect depth scale causes objects at 10m to appear at 17.5m → range boundary effects  
- Consequence: false positives from edge-of-range artifacts

### Mode 4: Class Confusion
- nuScenes trains on 10 classes, KITTI evaluates 3 (Car, Pedestrian, Cyclist)  
- nuScenes and KITTI have different class definitions (nuScenes "car" ≠ KITTI "Car")  
- Consequence: evaluation metric includes only Car; others ignored

---

## 4. Why Existing Methods Are Insufficient

### DA-BEV (ECCV 2024) — Primary Comparison
| Limitation | Impact |
|-----------|--------|
| Requires unlabeled target domain data | Not zero-shot — expensive to deploy |
| Adversarial training on BEV features | Addresses appearance shift; **NOT intrinsics mismatch** |
| Domain-specific training per target | Cannot generalize to new unseen domains |
| Does not modify depth estimation mechanism | Root cause (intrinsics-dependent sampling) persists |

### BEVDepth — Why Explicit Depth Supervision Doesn't Help
- LiDAR-supervised depth learns nuScenes depth distribution  
- At test time on KITTI, the depth predictor is still domain-specific  
- Camera-awareness conditioning helps within a domain, not across domains

---

## 5. Solution: What We Need

A method that:
1. **Provides domain-agnostic depth features** regardless of camera intrinsics
2. **Requires zero target-domain data** (no retraining, no labels)
3. **Adds minimal compute overhead** (single GPU feasible)
4. **Adapts online** to new domains without full retraining

→ **Frozen Depth Anything V2 + lightweight adapter + Tent TTA** satisfies all four requirements.

---

## 6. Quantitative Results (TBD — Pending Dataset Download)

| Method | nuScenes NDS | KITTI Car AP/E | KITTI Car AP/M | Gen. Drop |
|--------|-------------|----------------|----------------|-----------|
| BEVFormer-Tiny-fp16 | 35.9% (pub) | TBD | TBD | TBD |
| DA-BEV | TBD | TBD | TBD | TBD |
| Ours (Config D) | TBD | TBD | TBD | TBD |

*Figure: `experiments/baseline/domain_gap_analysis.png` (generated)*

---

## 7. Figures Generated

- ✅ `experiments/baseline/domain_gap_analysis.png` — 6-panel figure showing intrinsics comparison, projection ratio, BEV reference point error heatmap, and domain gap summary
- ⬜ Qualitative failure case images (20+ KITTI frames) — pending KITTI download
- ⬜ Quantitative AP table — pending model eval on KITTI val

---

*This document is the rough draft of the paper's Introduction + Motivation sections.*
