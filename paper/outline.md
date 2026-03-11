# Paper Outline & Draft
# Domain-Generalizable Camera-Only BEV 3D Detection via Foundation Depth Priors

**Working title:** *"DepthPrior-BEV: Domain-Generalizable 3D Detection via Frozen Foundation Depth Features"*  
**Target venue:** CVPR 2026 (submission deadline: November 2025)  
**Status:** Literature/method complete → experiments pending dataset download

---

## Abstract (Draft)

Camera-only BEV 3D object detection methods achieve strong performance on their training domain but suffer severe degradation when deployed in new environments. We identify camera intrinsic mismatch as the root cause: when focal lengths differ by 1.75× (nuScenes vs KITTI), BEVFormer's spatial cross-attention samples image features at positions offset by up to 379 pixels, fundamentally breaking depth estimation. We propose to address this via frozen foundation depth priors. Specifically, we inject frozen Depth Anything V2 (ViT-S) features, combined with intrinsics normalization and a lightweight 952K-parameter adapter, into BEVFormer's feature extraction pipeline. Additionally, we apply test-time adaptation (Tent) on adapter BatchNorm parameters to further reduce distribution shift at inference. Our method achieves X% cross-domain transfer improvement on KITTI while maintaining in-domain nuScenes performance, without requiring any target-domain training data.

---

## 1. Introduction

### 1.1 Problem Statement
- Autonomous driving deployment requires generalization across vehicles, cities, and countries
- Current camera-only BEV methods (BEVFormer, BEVDepth, StreamPETR) drop significantly
- Root cause: camera intrinsics (focal length, principal point) are different across domains

### 1.2 Key Insight
- Existing domain adaptation (DA-BEV) requires target-domain data → impractical
- Foundation models (Depth Anything V2) are trained on 62M diverse images → domain-agnostic
- Frozen DAv2 encoder provides geometry-consistent features regardless of camera calibration

### 1.3 Contributions
1. We identify and quantify the intrinsic mismatch problem (379–13832px reference point error)
2. We propose intrinsics normalization to canonicalize images before foundation depth extraction
3. We propose DepthPrior-BEV: frozen DAv2 features + lightweight adapter injected into BEVFormer
4. We propose TTA via entropy minimization on adapter BN params (zero target labels)
5. We establish the first cross-domain camera-only BEV benchmark with nuScenes→KITTI protocol

---

## 2. Related Work

### 2.1 Camera-Only BEV Detection
- BEVFormer [Li et al., ECCV 2022]: spatial cross-attention BEV queries (our baseline)
- BEVDepth [Li et al., AAAI 2023]: explicit depth supervision → limited cross-domain
- StreamPETR [Wang et al., ICCV 2023]: object-centric temporal modeling
- DETR3D [Wang et al., CoRL 2021]: sparse query-based detection

### 2.2 Domain Adaptation for 3D Detection
- DA-BEV [Jiang et al., ECCV 2024]: adversarial alignment of BEV features
- Limitations: requires target data, doesn't solve intrinsics mismatch

### 2.3 Foundation Depth Models
- Depth Anything V2 [Yang et al., NeurIPS 2024]: synthetic pre-training + 62M pseudo-labels
- Key property: zero-shot generalization across unseen domains
- We use FEATURES from frozen encoder, not depth predictions

### 2.4 Test-Time Adaptation
- Tent [Wang et al., ICLR 2021]: entropy minimization on BatchNorm params
- We adapt only adapter BatchNorm (~2K params) — targeted, stable, fast

---

## 3. Method: DepthPrior-BEV

### 3.1 Preliminary: BEVFormer's Depth Estimation Bottleneck
- Spatial Cross-Attention (SCA) projects BEV queries to 2D using intrinsics K
- Intrinsics are implicit in reference point projection → domain-specific
- Fig. 1: Cross-domain failure mechanism (reference point error heatmap)

### 3.2 Intrinsics Normalization
- Warp input image via: H = K_canonical @ inv(K_source)
- Maps any camera to canonical view (nuScenes CAM_FRONT)
- Makes downstream features intrinsics-independent (Fig. 2)

### 3.3 Depth Feature Extraction (Frozen DAv2-ViTS)
- DINOv2-ViT-S encoder from Depth Anything V2 (frozen, 24.8M params)
- Extract patch features from final encoder layer: B × (H/14) × (W/14) × 384
- No depth prediction used — only intermediate features

### 3.4 Lightweight Feature Adapter
- 2-layer ConvNet (3×3 conv + BN + ReLU + 1×1 conv + BN)
- Maps 384D DAv2 features → 256D BEVFormer feature space
- ~952K trainable parameters total

### 3.5 Feature Injection into BEVFormer
- Injection point: after img_neck (FPN output), before BEV encoder
- Fusion: img_feats = backbone_feats + α × adapter(depth_feats)
- α initialized to 0.1, learned to ~0.X during training
- Non-invasive: BEV encoder, TSA, SCA unchanged

### 3.6 Test-Time Adaptation (Tent)
- At test time on target domain (KITTI):
  - Set adapter BN layers to train mode
  - Minimize entropy H(p) = -Σ pᵢ log pᵢ of detection scores
  - Update only adapter BN params (γ, β): ~2K params
  - Reset at start of each new sequence
- Hyperparameters: lr=1e-4, steps=1 (tuned on 50-sample subset)

---

## 4. Experiments

### 4.1 Datasets and Metrics
**Training:** nuScenes v1.0-trainval (700 scenes)  
**In-domain eval:** nuScenes val (150 scenes) — NDS, mAP  
**Cross-domain eval:** KITTI 3D val (3769 frames) — Car 3D AP @ IoU=0.7, Easy/Moderate/Hard  

### 4.2 Implementation Details
- Base: BEVFormer-Tiny-fp16 (ResNet-50, 6.5GB VRAM, BEV=50×50)
- GPU: NVIDIA RTX 5060 Ti (16GB)
- Training: AdamW, lr=2e-4, 24 epochs, cosine schedule, fp16
- DAv2 checkpoint: depth_anything_v2_vits.pth (94.6MB)

### 4.3 Main Results (Table 1)

| Config | Method | nuScenes NDS | nuScenes mAP | KITTI AP/E | KITTI AP/M | Gen. Drop |
|--------|--------|-------------|-------------|-----------|-----------|-----------|
| A | BEVFormer-Tiny-fp16 (baseline) | TBD | TBD | TBD | TBD | — |
| B | + DAv2 (random adapter) | TBD | TBD | TBD | TBD | TBD |
| C | + DAv2 + Trained Adapter | TBD | TBD | TBD | TBD | TBD |
| D | + DAv2 + Adapter + TTA (Ours) | TBD | TBD | TBD | TBD | TBD |
| — | DA-BEV (ECCV 2024) | TBD | TBD | TBD | TBD | TBD |

### 4.4 Ablation Study (Table 2)

| Config | Variant | KITTI AP/E | Δ vs Ours |
|--------|---------|-----------|---------|
| D | Full method | TBD | — |
| E | Depth values (not features) from DAv2 | TBD | TBD |
| F | Without intrinsics normalization | TBD | TBD |
| G | Random adapter (no DAv2) | TBD | TBD |
| H | DAv2-Large (ViT-L) | TBD | TBD |

### 4.5 Efficiency Comparison (Table 3)

| Method | Trainable Params | Extra VRAM | Latency Overhead |
|--------|-----------------|-----------|-----------------|
| BEVFormer-Tiny-fp16 | 30.6M | — | — |
| Ours (Config D) | 952K adapter | +0.17 GB | TBD ms |

---

## 5. Analysis

### 5.1 Feature Visualization
- t-SNE of depth features: nuScenes vs KITTI should cluster similarly (proving domain-agnostic)
- BEV feature comparison: before/after depth prior injection

### 5.2 TTA Dynamics
- α (depth_scale) evolution during training
- BN param drift during TTA: confirms adaptation is happening
- Entropy curves: decreasing during TTA sequences

### 5.3 Failure Case Analysis
- Cases where our method still fails: extreme weather, night, heavily occluded objects
- Class-specific analysis: pedestrians likely harder than cars

---

## 6. Conclusion

We presented DepthPrior-BEV, a domain-generalizable camera-only BEV 3D detection framework that achieves strong cross-domain transfer without any target-domain data. By combining intrinsics normalization, frozen foundation depth features, a lightweight adapter, and targeted test-time adaptation, we significantly improve KITTI performance of a nuScenes-trained model while maintaining in-domain performance.

---

## Figures Planned

1. Architecture overview diagram (backbone → DAv2 → adapter → BEVFormer encoder)
2. Domain gap visualization (reference point error heatmap) — **DONE** (`domain_gap_analysis.png`)
3. Intrinsics normalization qualitative example
4. Main results bar chart (nuScenes NDS + KITTI AP)
5. Ablation study bar chart
6. Qualitative KITTI detections (baseline vs ours)
7. TTA dynamics curves

---

*Paper draft started: 2026-03-11*
