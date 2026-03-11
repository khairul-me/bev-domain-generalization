# Literature Notes: Domain-Generalizable BEV 3D Detection Research

**Project:** Domain-Generalizable Camera-Only BEV 3D Detection via Foundation Depth Priors  
**Date:** 2026-03-11  
**Researcher:** Solo

---

## Paper 1: BEVFormer (Priority 1)

**Citation:** Li et al., ECCV 2022 | arXiv:2203.17270

**Problem:** Multi-camera 3D perception for autonomous driving without LiDAR — how to build a unified BEV feature space from 2D camera images.

**Method (depth handling):**
- Predefined grid-shaped BEV queries (H×W grid covering the ground plane)
- **Spatial Cross-Attention (SCA):** Each BEV query projects 3D reference points onto 2D image planes using camera intrinsics/extrinsics, then samples image features via deformable attention
- **Temporal Self-Attention (TSA):** Warps previous BEV features into current frame using ego-motion and fuses them
- Depth is **implicitly** encoded by the projection from BEV space to image space — the network must *learn* which depth to attend to from the image features alone
- No explicit depth supervision during training (unlike BEVDepth)

**Key Architecture Detail:**
```
Input cameras → ResNet-101 + FPN backbone → image features
↓
BEV Queries (H×W×C) — learned grid
↓  (Spatial Cross-Attention via learned 3D→2D projection)
BEV Features → Detection Head (DETR-style)
```

**Results:**
- nuScenes val: 41.6% mAP, 51.7% NDS (BEVFormer-Base, 24ep)
- nuScenes test: 56.9% NDS (camera-only SOTA at the time)

**Limitation:**
- Spatial cross-attention relies on camera intrinsics being **consistent with training distribution**
- No explicit depth — network must implicitly learn depth from appearance, making it brittle under domain shift
- High VRAM: BEVFormer-Base requires ~28.5GB (use tiny/small for RTX 5060)

**Relevance to my work:**
- This is the baseline I modify. The spatial cross-attention's implicit depth mechanism is exactly what I replace/augment with DAv2 features
- The 3D reference point projection step is where I inject depth prior features
- Note: BEVFormer-Tiny (R50) uses only 6.5GB — **use Tiny variant for RTX 5060 Ti**

---

## Paper 2: BEVDepth (Priority 2)

**Citation:** Li et al., AAAI 2023 | arXiv:2206.10092

**Problem:** Depth estimation in existing BEV methods is unreliable because it relies on the network implicitly learning depth from appearance alone.

**Method (depth handling):**
- Introduces **explicit depth supervision**: projects LiDAR points onto image plane and uses them as ground-truth depth labels
- **Camera-awareness depth estimation module**: conditions the depth network on camera intrinsics (fx, fy, cx, cy) — this helps with different camera configurations
- **Depth Refinement Module (DRM):** corrects errors from imprecise feature unprojection (voxel pooling artifacts)
- **Efficient Voxel Pooling:** custom CUDA kernel for fast BEV feature construction from depth-lifted image features

**Results:**
- nuScenes test: 60.9% NDS — first camera model to reach 60% NDS

**Limitation:**
- Explicit depth supervision from LiDAR is intrinsic to this method — applying to another domain (KITTI) requires re-generating LiDAR pseudo-labels
- The depth estimation network still depends on learned visual appearance for regions without LiDAR points
- Camera-awareness module helps within a domain, but the depth distribution shift across domains (different focal lengths) is not fully resolved

**Relevance to my work:**
- **Core motivation source**: BEVDepth proves depth is the bottleneck. If even their "explicit depth" supervision fails cross-domain, it proves my approach (frozen foundation model) is the right direction
- The camera-awareness intrinsics conditioning is complementary to my intrinsics normalization idea

---

## Paper 3: StreamPETR (Priority 3)

**Citation:** Wang et al., ICCV 2023 | arXiv:2303.11926

**Problem:** Temporal modeling in multi-view 3D detection requires high computation overhead for long sequence memory.

**Method (depth handling):**
- Object-centric temporal mechanism: propagates information through sparse object queries (not dense BEV grids)
- Each object query stores a 3D position + feature vector, updated frame-by-frame
- **Motion-aware Layer Normalization:** compensates for object movement between frames
- Depth: implicit, relies on PETR-style positional encoding where 3D rays from camera are encoded into image features

**Results:**
- nuScenes val: 67.6% NDS (camera-only) — comparable to LiDAR-based methods

**Limitation:**
- Positional encoding uses absolute camera intrinsics — if intrinsics change (cross-domain), 3D position encoding is incorrect
- Object propagation assumes consistent frame rate and ego-motion — different datasets have different conventions

**Relevance to my work:**
- Secondary baseline to optionally replicate
- Shows that even SOTA camera-only methods are fundamentally tied to their training domain's camera calibration
- The intrinsics-embedded positional encoding is another failure mode worth documenting in Task 6

---

## Paper 4: DETR3D (Priority 4)

**Citation:** Wang et al., CoRL 2021 | arXiv:2110.06922

**Problem:** 3D object detection from multi-view cameras without explicit depth prediction.

**Method (depth handling):**
- Sparse 3D object queries initialized in 3D space
- Each query projects its 3D position onto each camera image plane to sample features
- No per-pixel depth estimation — operates directly in 3D with learned queries
- Key claim: "top-down approach outperforms bottom-up (depth-pred first) because it avoids compounding depth prediction error"

**Results:**
- nuScenes test: 41.2% NDS (camera-only, significantly below BEVFormer successor)

**Limitation:**
- Cannot model scene-level structure (only objects) — no BEV map
- Still implicitly uses camera intrinsics for 3D→2D projection

**Relevance to my work:**
- Foundational predecessor to BEVFormer and StreamPETR
- Establishes the "query-based" paradigm for camera-only 3D detection
- Related work citation in the paper

---

## Paper 5: Depth Anything V2 (Priority 5)

**Citation:** Yang et al., NeurIPS 2024 | arXiv:2406.09414

**Problem:** Monocular depth estimation that generalizes across diverse real-world domains.

**Method:**
- Key insight: replace all labeled real images with **synthetic images** (fine-grained, precise labels)
- Scale up teacher model (larger capacity ViT), then use it to generate pseudo-labels for 62M unlabeled real images
- Student models (ViT-S, ViT-B, ViT-L, ViT-G) trained on both synthetic + pseudo-labeled real images
- **Architecture**: Dense Prediction Transformer (DPT) with ViT encoder
- Training: 595K synthetic + 62M pseudo-labeled real images

**Model Variants:**
| Model | Params | Speed | Channels |
|-------|--------|-------|----------|
| ViT-S | 25M | Fast | 64 intermediate, [48, 96, 192, 384] output |
| ViT-B | 97M | Medium | 128 intermediate, [96, 192, 384, 768] output |
| ViT-L | 335M | Slow | 256 intermediate, [256, 512, 1024, 1024] output |

**Results:**
- Significantly better depth quality and zero-shot generalization vs V1
- 10x+ faster than Stable Diffusion-based depth methods

**Limitation:**
- Produces relative depth (not metric) — but this is actually fine for our use case since we use intermediate *features*, not depth values

**Relevance to my work:**
- **Core component**: I use the frozen ViT-S encoder (25M params, fast) as a geometric prior
- The 62M diverse training images make it domain-agnostic — exactly what we need for cross-domain transfer
- I extract intermediate features from the DPT encoder (not the final depth output) to inject into BEVFormer
- **Critical**: keep fully frozen to preserve domain generalization

---

## Paper 6: DA-BEV (Priority 6)

**Citation:** Jiang et al., ECCV 2024 | arXiv:2401.08687

**Problem:** Unsupervised domain adaptation for camera-only BEV perception (3D detection + segmentation).

**Method:**
- First domain-adaptive camera-only BEV framework
- **Query-based Adversarial Learning (QAL):** domain discriminator operates on BEV queries, adversarially aligned between source and target
- **Query-based Self-Training (QST):** pseudo-labels generated from image-view features, used to train BEV features
- Exploits complementary relationship between image-view and BEV features

**Results:**
- Superior cross-domain performance on: weather changes, illumination, city configurations

**Limitation:**
- **Requires unlabeled target domain data** for adversarial training — not zero-shot
- Domain-specific training: if target domain changes, must re-run adaptation
- Adversarial training is unstable — requires careful tuning
- Does not address fundamental intrinsic parameter mismatch (mainly handles appearance shift)

**Relevance to my work:**
- **Primary comparison method** in my experiments
- My method's key advantage over DA-BEV: **completely zero-shot** — no target data, no re-training
- DA-BEV's requirement for unlabeled target data is a major practical limitation I solve

---

## Paper 7: OpenAD Benchmark (Priority 7)

**Citation:** Zhou et al., NeurIPS 2024 | arXiv:2411.17761

**Problem:** No standardized benchmark for evaluating generalization in autonomous driving 3D detection.

**Method:**
- First real-world open-world autonomous driving benchmark for 3D detection
- Unifies 5 existing AD datasets (nuScenes, KITTI, Waymo, ONCE, Argoverse 2)
- 2,000 scenes with corner case annotations
- Evaluates: scene generalization, cross-vehicle-type, open-vocabulary, corner case detection

**Key Findings:**
- Models trained on one dataset generalize poorly to others (especially geometry-dependent tasks)
- Domain gap is largest when camera intrinsics differ significantly

**Relevance to my work:**
- Validates that cross-domain degradation is a real, documented problem
- Provides a third evaluation benchmark beyond KITTI (Argoverse 2) for Task 11
- Use their evaluation protocol in Task 6 for failure documentation

---

## Paper 8: nuScenes Dataset Paper (Priority 8)

**Citation:** Caesar et al., CVPR 2020 | arXiv:1929.12008 (correct ID: 1929.12208 → actually arXiv:2008.01308)

**Key Details:**
- Boston + Singapore locations, 6 cameras (360° coverage), 32-beam LiDAR
- 1000 scenes (700 train, 150 val, 150 test), ~1.4M camera images
- Camera intrinsics: focal length ~1266px (for 1600×900 images), horizontal FoV ~60°

**NDS Metric Formula:**
```
NDS = (1/10) × [5×mAP + Σ(1 - min(1, mTPₑ))]
```
where TP errors: mATE (translation), mASE (scale=1-3D_IoU), mAOE (orientation), mAVE (velocity), mAAE (attribute)

- mAP uses center-distance-based matching (0.5m, 1m, 2m, 4m thresholds), averaged over 10 classes and 4 distance thresholds
- **10 detection classes**: car, truck, bus, trailer, construction_vehicle, pedestrian, motorcycle, bicycle, traffic_cone, barrier

**Relevance to my work:**
- Primary training and in-domain evaluation benchmark
- NDS formula understanding is critical for correctly reading results tables
- Camera intrinsics (fx≈1266, 1600×900) serve as the "canonical" camera for intrinsics normalization in Task 7.4

---

## Paper 9: KITTI Benchmark (Priority 9)

**Citation:** Geiger et al., CVPR 2012

**Key Details:**
- Karlsruhe, Germany, stereo cameras, 64-beam Velodyne
- Training: 7,481 labeled frames | Test: 7,518 unlabeled
- Camera intrinsics: focal length ~721px (for 1242×375 images) — **roughly half of nuScenes focal length**
- 3 classes: Car, Pedestrian, Cyclist
- Evaluation sections: Easy/Moderate/Hard based on object height and occlusion

**3D AP Evaluation:**
- IoU=0.7 for Car, IoU=0.5 for Pedestrian/Cyclist
- Uses 11-point or 40-point interpolation for AP curve
- Converts BEV boxes to 3D for evaluation

**Domain Gap vs nuScenes:**
| Property | nuScenes | KITTI | Impact |
|----------|----------|-------|--------|
| Focal length | ~1266px | ~721px | 1.75× difference → depth scale |
| Image size | 1600×900 | 1242×375 | Different resolution |
| LiDAR beams | 32 | 64 | More precise depth supervision |
| Ego height | SUV (~1.8m) | Sedan (~1.4m) | Camera height offset |
| Environment | Urban (Boston/SG) | Suburban (Karlsruhe) | Appearance shift |

**Relevance to my work:**
- Primary cross-domain evaluation target (Task 6, 11)
- The **~1.75× focal length difference is the key quantitative driver** of depth distribution shift

---

## Paper 10: Tent — Test-Time Adaptation (Priority 10)

**Citation:** Wang et al., ICLR 2021 | arXiv:2006.10726

**Problem:** Model must generalize to new data at test time without any source data or labels.

**Method:**
- Adapt only **BatchNorm affine parameters** (scale γ, shift β) at test time
- Loss: entropy of model predictions: `H(p) = -Σ pᵢ log(pᵢ)`
- Update with SGD/Adam on entropy loss, applied to each batch independently
- Key insight: BN statistics capture domain-specific distribution; adapting them is cheap and effective

**Algorithm:**
```
For each test batch:
  1. Forward pass with current γ, β
  2. Compute entropy of output predictions
  3. Backward on entropy loss
  4. Update only γ, β (all other params frozen)
  5. Return predictions from step 1
```

**Results:**
- ImageNet-C: state-of-the-art among test-time methods
- Digit recognition (SVHN→MNIST): source-free DA
- Semantic segmentation (GTA→Cityscapes): competitive

**Limitation:**
- Sensitive to batch size (needs sufficient diversity for stable BN statistics)
- Can drift if domain keeps changing within a test sequence
- Steps>1 per sample helps but risks overfitting to individual samples

**Relevance to my work:**
- **Directly implement this** in Task 9
- Apply to BN layers in the adapter module only (lightweight, targeted)
- Reset γ, β to trained values before each KITTI test sequence
- Tune steps ∈ {1, 3, 5} and lr ∈ {1e-5, 1e-4, 1e-3}

---

## Step 2.6: Depth Estimation Chain in BEVFormer

### Data Flow Diagram

```
Raw Camera Images [6×3×H×W]
         │
         ▼
  ResNet + FPN Backbone
  (6 cameras processed independently)
  Output: multi-scale feature maps [6 × C × H/4,H/8,H/16,H/32 × W/...]
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │         BEVFormer Encoder (6 layers)                │
  │                                                     │
  │  BEV Queries Q [H_bev × W_bev × C]                 │
  │         │                                           │
  │         ▼                                           │
  │  Temporal Self-Attention (TSA)                      │
  │  (fuse with prev BEV features + ego-warp)           │
  │         │                                           │
  │         ▼                                           │
  │  Spatial Cross-Attention (SCA)   ←── THIS IS WHERE  │
  │  ┌────────────────────────────┐       DEPTH LIVES   │
  │  │ For each BEV query q at   │                     │
  │  │ (x,y) on ground plane:    │                     │
  │  │   1. Sample ref points in │                     │
  │  │      column: z ∈ [0, Z_max]│              ◄────  │
  │  │   2. Project (x,y,z) → 2D │ ← Camera intrinsics │
  │  │      image coord per cam  │   K, extrinsics T    │
  │  │   3. Deformable attention  │                     │
  │  │      to sample features   │                     │
  │  └────────────────────────────┘                     │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  BEV Features [H_bev × W_bev × C]
         │
         ▼
  Detection Head (DETR-style)
  → 3D bounding boxes, classes, velocities
```

### Where Depth Enters
1. **Step 2 in SCA**: when projecting 3D reference points at heights z={z₁, z₂, ..., zₙ} onto image planes using camera intrinsics K
2. The network implicitly learns to attend to image regions that encode depth via appearance (shadows, size, blur)
3. **Intrinsics assumption**: K_nuScenes (fx≈1266) coded into the projection matrices — when K_KITTI (fx≈721) is used, the projected reference points land at wrong 2D coordinates, causing the model to sample wrong image regions

### Why This Causes Cross-Domain Failure
- Same 3D point at (x=5m, y=0, z=1.5m) projects to different image coordinates in nuScenes vs KITTI
- BEV queries expect features from nuScenes image coordinates → sample wrong regions in KITTI images  
- The depth "prior" embedded in the network is nuScenes-specific

### Where I Inject DAv2 Features (Task 7)
- **After the backbone, before SCA**: inject depth features into the image feature maps
- DAv2 provides geometry-aware features that represent the scene's depth structure
- These features augment the backbone features, giving the SCA module reliable depth cues regardless of intrinsics
- The intrinsics normalization (Task 7.4) ensures DAv2 sees a canonical view

---

## Summary: Key Design Decisions for My Method

| Design Choice | Rationale from Literature |
|---------------|--------------------------|
| Freeze DAv2 completely | Maintains domain-agnostic depth prior from 62M diverse training images |
| Use ViT-S (not ViT-B/L) | Fits in RTX 5060 Ti VRAM budget; still sufficient for depth features |
| Inject after backbone | Cleanest integration point; preserves BEVFormer's existing temporal features |
| Intrinsics normalization | Allows DAv2 to produce consistent features regardless of source camera |
| TTA via BN adaptation (Tent) | Zero-label, online, lightweight — only adapter BN params; Tent-proven approach |
| Adapter ~1-5M params only | Feasible to train on single GPU; strong constraint → strong regularization |

---

*Notes completed: 2026-03-11*
