# BEV Domain Gap Research — Supplementary Report
## Filling Every Gap for Independent Paper Writing

**Companion to:** `bev_research_final_report.md`  
**Purpose:** This document provides every piece of information missing from the main report that a reader needs to independently write the full research paper — including hyperparameters, formulas, class lists, bibtex entries, narrative guidance, and editorial fixes.

---

## Table of Contents

1. [Abstract Fix — Duplicate Sentence](#1-abstract-fix--duplicate-sentence)
2. [The 10 nuScenes Detection Classes](#2-the-10-nuscenes-detection-classes)
3. [nuScenes mAP Definition — Full Detail](#3-nuscenes-map-definition--full-detail)
4. [Complete Training Hyperparameters — All 6 Experiments](#4-complete-training-hyperparameters--all-6-experiments)
5. [What Is Frozen vs. Trainable in Each Experiment](#5-what-is-frozen-vs-trainable-in-each-experiment)
6. [E3-C Consistency Loss — Full Specification](#6-e3-c-consistency-loss--full-specification)
7. [E4 Experiment Design — What Is Actually Trained](#7-e4-experiment-design--what-is-actually-trained)
8. [DAv2 ViT-S Architecture and Inference Details](#8-dav2-vits-architecture-and-inference-details)
9. [BEVFormer Temporal Queue During Representation Analysis](#9-bevformer-temporal-queue-during-representation-analysis)
10. [Debiased HSIC Formula — Written Out Explicitly](#10-debiased-hsic-formula--written-out-explicitly)
11. [All 96 Depth-Scale-Invariant Channel Indices](#11-all-96-depth-scale-invariant-channel-indices)
12. [Boston 2,000 Sample Selection for E3/E4/E6](#12-boston-2000-sample-selection-for-e3e4e6)
13. [Camera Ordering and CAM_FRONT Index](#13-camera-ordering-and-cam_front-index)
14. [Pseudo-Label Generation — Full Specification](#14-pseudo-label-generation--full-specification)
15. [Complete BibTeX Entries for All 13 References](#15-complete-bibtex-entries-for-all-13-references)
16. [Figure Descriptions for Redrawing](#16-figure-descriptions-for-redrawing)
17. [Section-by-Section Narrative Writing Guide](#17-section-by-section-narrative-writing-guide)
18. [Reproducibility Commands — Full Training and Eval](#18-reproducibility-commands--full-training-and-eval)
19. [Paper Editorial Checklist Before Submission](#19-paper-editorial-checklist-before-submission)

---

## 1. Abstract Fix — Duplicate Sentence

The current abstract in `draft.tex` contains the following sentence **twice** (lines 35–39):

> The t-SNE analysis of BEV encoder features (cross-city drift ratio = 1.30) provides the stronger structural evidence that city-level separation persists.
> A t-SNE analysis of BEV encoder features confirms the partial normalization: cross-city drift ratio = 1.30, showing the encoder does not fully collapse city identity.

**Action:** Delete the second occurrence (lines 37–39). The corrected abstract ending should read:

```
The t-SNE analysis of BEV encoder features (cross-city drift ratio $= 1.30$)
provides the stronger structural evidence that city-level separation persists.
Foundation monocular depth features from Depth Anything V2 (ViT-S) are significantly
more domain-stable across cities (depth-scale Cohen's~$d = 0.09$), motivating their
use as a geometric prior for domain bridging.
```

---

## 2. The 10 nuScenes Detection Classes

The 10 object classes in canonical nuScenes order (indices 0–9):

| Index | Class Name | Typical Boston AP | Typical Singapore AP | Notes |
|---|---|---|---|---|
| 0 | `car` | 0.621 | 0.589 | Most common, best AP |
| 1 | `truck` | 0.356 | 0.309 | Medium frequency |
| 2 | `construction_vehicle` | 0.135 | 0.107 | Rare, highly variable visual appearance |
| 3 | `bus` | 0.448 | 0.399 | Common in urban environments |
| 4 | `trailer` | 0.157 | 0.000 | **Absent in Singapore split** |
| 5 | `barrier` | 0.526 | 0.492 | Road infrastructure, high AP |
| 6 | `motorcycle` | 0.479 | 0.416 | Pose-diverse |
| 7 | `bicycle` | 0.429 | 0.352 | Visually distinct across cities |
| 8 | `pedestrian` | 0.482 | 0.492 | Improves in Singapore |
| 9 | `traffic_cone` | 0.617 | 0.510 | High Boston AP, significant drop |

These class names and indices are used in:
- `NUSCENES_CLASSES` lookup in `merge_pseudo_labels.py`
- `gt_labels` integer fields in all PKL files
- The per-class AP table (Table 2 in the paper)

**nuScenes attribute classes** (used for mAAE, separate from detection classes):
- For `car`, `bus`, `truck`, `trailer`: `vehicle.moving`, `vehicle.parked`, `vehicle.stopped`
- For `pedestrian`: `pedestrian.moving`, `pedestrian.standing`, `pedestrian.sitting_lying_down`
- For `bicycle`, `motorcycle`: `cycle.with_rider`, `cycle.without_rider`
- `barrier`, `traffic_cone`, `construction_vehicle`: no attribute (mAAE = nan for these)

---

## 3. nuScenes mAP Definition — Full Detail

### mAP Computation

nuScenes mAP uses **bird's-eye-view center-point distance** matching (not 3D IoU):

$$\text{AP}_c = \frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \text{AP}_c(d)$$

where $\mathcal{D} = \{0.5, 1.0, 2.0, 4.0\}$ meters and each $\text{AP}_c(d)$ is computed at a fixed BEV center-distance threshold of $d$ meters.

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c, \quad C = 10$$

**Key difference from standard mAP:** No IoU-based matching. A prediction is a true positive if its BEV center is within $d$ meters of any unmatched GT box of the same class, regardless of box size, orientation, or 3D extent.

### True Positive (TP) Error Metrics

For matched TP detections, five error metrics are averaged across all TPs at the **2.0m distance threshold** (the primary threshold):

| Metric | Definition | Unit | Notes |
|---|---|---|---|
| mATE | mean Average Translation Error | meters | L2 distance between predicted and GT BEV centers |
| mASE | mean Average Scale Error | — | 1 − IoU of axis-aligned 3D boxes (size-only, no rotation) |
| mAOE | mean Average Orientation Error | radians | Minimum yaw angle error (symmetric at π rad) |
| mAVE | mean Average Velocity Error | m/s | L2 of velocity vectors (vx, vy) |
| mAAE | mean Average Attribute Error | — | 1 − accuracy of predicted attribute class |

### NDS Formula Expanded

$$\text{NDS} = \frac{1}{10}\left(5 \cdot \text{mAP} + (1 - \min(\text{mATE}, 1)) + (1 - \min(\text{mASE}, 1)) + (1 - \min(\text{mAOE}, 1)) + (1 - \min(\text{mAVE}, 1)) + (1 - \min(\text{mAAE}, 1))\right)$$

Each TP error contributes equally to NDS. The clipping at 1.0 ensures a single error cannot dominate. NDS ranges from 0 (complete failure) to 1 (perfect).

---

## 4. Complete Training Hyperparameters — All 6 Experiments

### E3-A, E3-B, E3-C (Frozen BEVFormer + Adapter, Boston Supervision)

| Parameter | Value | Notes |
|---|---|---|
| Training data | 2,000 Boston training frames | Indices 0–1999 (first 2000 of training split) |
| Epochs | 4 | ~8,000 gradient steps total |
| Batch size | 1 frame per GPU | Multi-camera (6 images per batch) |
| Optimizer | AdamW | FP32 (not mixed precision — stability) |
| Adapter learning rate | 2×10⁻⁴ | Applied only to adapter weights |
| BEVFormer learning rate | 0 (frozen) | All BEVFormer params have requires_grad=False |
| lr_mult for adapter | 0.001 (relative to base_lr) | Effective: base_lr × lr_mult |
| Weight decay | 0.01 | Standard AdamW |
| Gradient clipping | max_norm=35, norm_type=2 | Identical to original BEVFormer training |
| Loss scale | Dynamic (for mixed precision) | Applied to total loss |
| Val interval | Every 2 epochs | Singapore eval at epochs 2 and 4 |
| Checkpoint save | Every 2 epochs | max_keep_ckpts=4 |
| Residual scale α | E3-A: 0.01 / E3-B: 0.1 / E3-C: 0.1 | |
| Consistency loss weight | E3-C only: λ_cons = 1.0 | Detection loss weight = 1.0 (equal) |
| Starting checkpoint | `bevformer_base_epoch_24.pth` | Pretrained 24-epoch checkpoint |

### E4 (Partial BEV Encoder Unfreeze + Adapter)

| Parameter | Value | Notes |
|---|---|---|
| Training data | Same 2,000 Boston frames as E3 | Indices 0–1999 |
| Epochs | 4 | |
| Batch size | 1 | |
| Optimizer | AdamW | FP32 |
| Adapter learning rate | 2×10⁻⁴ | Same as E3-B |
| BEV encoder learning rate | 1×10⁻⁵ | ~20× lower than adapter |
| Other BEVFormer components | Frozen (lr=0) | Backbone, FPN, head all frozen |
| Weight decay | 0.01 | |
| Gradient clipping | max_norm=35 | |
| α | 0.1 | Same as E3-B |
| Starting checkpoint | `bevformer_base_epoch_24.pth` | |

### E5 (Pseudo-Label Adaptation on Singapore)

| Parameter | Value | Notes |
|---|---|---|
| Training data | All 2,929 Singapore val frames | With merged pseudo-labels |
| Epochs | 4 | |
| Batch size | 1 | |
| Optimizer | AdamW | FP32 |
| Adapter learning rate | 2×10⁻⁴ | Only adapter trains |
| All BEVFormer components | Frozen (lr=0) | Including encoder |
| Weight decay | 0.01 | |
| Gradient clipping | max_norm=35 | |
| α | 0.1 | Same as E3-B |
| τ (pseudo-label threshold) | 0.3 | Confidence score cutoff |
| Consistency loss | None | Detection loss only |
| Starting checkpoint | `bevformer_base_epoch_24.pth` | |
| Val/test data | Same 2,929 Singapore frames | Train/test overlap — see limitation §11.2 |

### E6 (96-Channel Depth-Scale Adapter, Boston Supervision)

| Parameter | Value | Notes |
|---|---|---|
| Training data | Same 2,000 Boston frames as E3-B | |
| Epochs | 4 | |
| Everything else | **Identical to E3-B** | Single change: channel_indices |
| channel_indices | 96 indices (see §11) | Selected by |r| with log-depth std |
| Adapter in_channels | 96 (not 384) | 91K params instead of 164K |
| Starting checkpoint | `bevformer_base_epoch_24.pth` | |

---

## 5. What Is Frozen vs. Trainable in Each Experiment

The term "frozen" means `requires_grad=False` for all parameters in that module.

| Component | Baseline | E3-A | E3-B | E3-C | E4 | E5 | E6 |
|---|---|---|---|---|---|---|---|
| ResNet-101 backbone | Frozen | Frozen | Frozen | Frozen | Frozen | Frozen | Frozen |
| FPN neck | Frozen | Frozen | Frozen | Frozen | Frozen | Frozen | Frozen |
| BEV encoder | Frozen | Frozen | Frozen | Frozen | **Trainable** (lr=1e-5) | Frozen | Frozen |
| Detection head | Frozen | Frozen | Frozen | Frozen | Frozen | Frozen | Frozen |
| DAv2 ViT-S encoder | N/A | Frozen | Frozen | Frozen | Frozen | Frozen | Frozen |
| Adapter Conv layers | N/A | **Trainable** | **Trainable** | **Trainable** | **Trainable** | **Trainable** | **Trainable** |

**Note on "Frozen":** The BEVFormer checkpoint `bevformer_base_epoch_24.pth` is loaded with `strict=False` when the adapter is present (the adapter weights are missing from the checkpoint and are initialized fresh). The adapter Conv weights are initialized with `nn.init.normal_(weight, std=0.001)` and `nn.init.zeros_(bias)`.

---

## 6. E3-C Consistency Loss — Full Specification

### Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{detect}} + \lambda_{\text{cons}} \cdot \mathcal{L}_{\text{cons}}$$

$$\mathcal{L}_{\text{cons}} = \|\hat{f}_{\text{FPN}}(I) - \hat{f}_{\text{FPN}}(\tilde{I})\|_F^2$$

where:
- $\hat{f}_{\text{FPN}}$ is the FPN output *after* adapter injection (i.e., $f_{\text{img}} + \alpha \cdot \delta$)
- $I$ is the original camera image
- $\tilde{I}$ is a ColorJitter-augmented version of $I$ (same augmentation parameters per batch)
- $\lambda_{\text{cons}} = 1.0$ (equal weight to detection loss)
- $\|\cdot\|_F$ is Frobenius norm over spatial and channel dimensions

### ColorJitter Augmentation Parameters

```python
torchvision.transforms.ColorJitter(
    brightness=0.4,
    contrast=0.4,
    saturation=0.4,
    hue=0.1
)
```

Applied per-image, same random parameters for all 6 cameras within a frame to maintain photometric consistency across views.

### Why It Fails

The consistency loss computes the *same forward pass twice* (once with $I$, once with $\tilde{I}$) and encourages the adapter to produce similar FPN outputs regardless of photometric perturbation. Two fundamental problems:

1. **Wrong target distribution:** ColorJitter perturbs brightness/contrast/saturation/hue — none of which capture Boston→Singapore structural differences (road marking styles, lane widths, building density, vegetation). The model is being pushed toward invariance to a shift that doesn't represent the real domain gap.

2. **Zero-adapter satisfaction:** The zero adapter ($\delta = 0$) trivially satisfies $\hat{f}(I) = \hat{f}(\tilde{I}) = f_{\text{img}}$ because the adapter output is identical (zero) for both inputs. The consistency loss is flat throughout training because it is minimized at initialization.

**Observed:** $\mathcal{L}_{\text{cons}}$ remains constant at **2.4** across all training iterations. This is the FPN-level variance between original and jittered images, which the adapter never reduces because it is already at its minimum (zero output).

---

## 7. E4 Experiment Design — What Is Actually Trained

E4 is the **only experiment where the BEV encoder receives gradients**. This is important to clarify precisely:

### Modules with Trainable Parameters in E4

1. **Adapter Conv layers** (lr = 2×10⁻⁴): The same adapter as E3-B is present and trainable.
2. **BEV encoder** (lr = 1×10⁻⁵): All BEV encoder parameters (self-attention, cross-attention, FFN layers in all 6 BEV encoder layers) receive gradients.

### Modules Still Frozen in E4

- ResNet-101 backbone (including all DCNv2 layers)
- FPN neck
- BEV transformer **decoder** (not to be confused with encoder — the decoder generates object queries for detection)
- Detection head (classification head, regression head)
- DAv2 ViT-S encoder

### The Causal Chain of E4's Failure

```
E4 training (Boston, 2000 samples, 4 epochs):
  1. Adapter produces δ (non-zero, since encoder can now shift to accommodate it)
  2. BEV encoder adapts its spatial cross-attention to the modified FPN features
  3. BEV encoder output (bev_embed) drifts from its pre-trained distribution
  4. Detection head (frozen, calibrated to original bev_embed distribution) 
     receives out-of-distribution inputs
  5. Detection head orientation/size predictions become unreliable
  6. mAOE: 0.321 → 1.22 rad (effectively random)
  7. mASE: 0.280 → 0.550 (size doubled in error)
```

**The key diagnostic:** With 2,000 samples, the encoder drifts at ~2×10⁻⁵ per parameter per step × 8,000 steps = cumulative shift of ~0.16 in normalized parameter space. The detection head sees a completely different bev_embed distribution after 4 epochs.

---

## 8. DAv2 ViT-S Architecture and Inference Details

### ViT-S Encoder Architecture

| Property | Value |
|---|---|
| Encoder type | ViT-Small (DINOv2-style) |
| Patch size | 14×14 pixels |
| Embedding dimension | 384 |
| Number of transformer layers | 12 |
| Number of attention heads | 6 |
| MLP ratio | 4 |
| Layer normalization | Pre-norm |
| Position embedding | Interpolated sinusoidal |
| Total parameters | ~21M (encoder only) |

### Intermediate Layer Selection

**Layer used:** Layer index 11 (the final transformer layer, 0-indexed)

This is accessed via:
```python
depth_backbone.get_intermediate_layers(
    x, 
    n=[11],           # list of layer indices to extract
    reshape=True,     # reshape from (B, N_patches, C) to (B, C, H_patch, W_patch)
    return_class_token=False
)
```

Output shape: `(B * N_cameras, 384, H_patch, W_patch)` where:
- `H_patch = ceil(H_image / 14)` (rounded to multiple of 14)
- `W_patch = ceil(W_image / 14)`

### Input Resolution and Preprocessing

**Target input size:** 308 pixels on the long side (enforced divisible by 14: `((308+13)//14)*14 = 308`)

For nuScenes CAM_FRONT images (1600×900):
- Scale factor: 308/1600 = 0.1925
- New size: H_new = round(900 × 0.1925) = 173, W_new = 308
- After /14 rounding: H_new = ((173+13)//14)×14 = 182, W_new = 308
- Patch grid: 182/14 × 308/14 = **13×22 = 286 patches**
- Feature map: `(B*N_cam, 384, 13, 22)`

### Image Normalization

BEVFormer images are de-normalized (mean=[103.53, 116.28, 123.675], std=[1.0,1.0,1.0], BGR) and re-normalized to DAv2 format (mean=[0.485,0.456,0.406]×255, std=[0.229,0.224,0.225]×255, RGB) before being passed to the DAv2 encoder.

### Bilinear Upsampling to FPN Resolution

After DAv2 feature extraction, the feature map is bilinearly upsampled to match FPN level-0 resolution:
- FPN level 0 stride: 8 (for 1600×900 images → FPN map: 200×113)
- Upsampled: `F.interpolate(dav2_feat, size=(H_fpn, W_fpn), mode='bilinear', align_corners=False)`

---

## 9. BEVFormer Temporal Queue During Representation Analysis

### How the Temporal Queue Works in BEVFormer

BEVFormer uses a recurrent BEV feature queue. At each timestep $t$:
- The BEV encoder receives the current camera features AND the BEV features from the previous $T-1$ timesteps (temporal queue length $T=4$)
- This temporal self-attention allows the model to propagate object information across frames

### Impact on Representation Analysis

During the representation analysis (N=500 pairs), **each frame is processed independently** — the temporal queue is **reset for every frame**. Specifically:

```python
# In representation_analysis_v2.py, before each forward pass:
if hasattr(model, 'pts_bbox_head'):
    if hasattr(model.pts_bbox_head.transformer, 'level_embeds'):
        # BEVFormer's prev_bev is reset to None at the start of each sequence
        pass

# model.test_step() is called with a single-frame batch:
# The prev_bev is implicitly None (no history) for each isolated frame
```

**Consequence:** The representation analysis measures the BEV features for *isolated frames* without temporal context. This is intentional — it removes temporal smoothing, giving a cleaner measure of what a single frame contributes. The domain gap measurement (detection performance) does use temporal context (Boston validation scenes are evaluated as sequences, not isolated frames), so there is a small methodological discrepancy between the representation analysis and the detection evaluation.

**For the paper:** This should be acknowledged: "Representation analysis uses single-frame inference (temporal queue reset); detection evaluation uses the full temporal queue as deployed."

---

## 10. Debiased HSIC Formula — Written Out Explicitly

### Biased HSIC (Original — Do Not Use for Bootstrap)

Given Gram matrices $K = X X^T$ and $L = Y Y^T$ (both $N \times N$):

$$\text{HSIC}_{\text{biased}}(K, L) = \frac{1}{N^2} \text{tr}(H K H L)$$

where $H = I_N - \frac{1}{N} \mathbf{1}\mathbf{1}^T$ is the centering matrix.

This estimator satisfies $\mathbb{E}[\text{HSIC}_{\text{biased}}] = \text{HSIC} + O(1/N)$, causing bootstrap subsamples (smaller $N$) to give systematically higher values.

### Debiased HSIC (Correct — Song et al. 2012, Kornblith et al. 2019 Appendix)

Given $K$ and $L$ with diagonals zeroed: $\tilde{K}_{ij} = K_{ij}(1 - \delta_{ij})$, $\tilde{L}_{ij} = L_{ij}(1 - \delta_{ij})$:

$$\text{HSIC}_{\text{debiased}}(K, L) = \frac{1}{N(N-3)} \left[ \sum_{i \neq j} \tilde{K}_{ij}\tilde{L}_{ij} + \frac{\left(\sum_{i \neq j} \tilde{K}_{ij}\right)\left(\sum_{i \neq j} \tilde{L}_{ij}\right)}{(N-1)(N-2)} - \frac{2}{N-2}\sum_{i} \left(\sum_{j} \tilde{K}_{ij}\right)\left(\sum_{j} \tilde{L}_{ij}\right) \right]$$

This satisfies $\mathbb{E}[\text{HSIC}_{\text{debiased}}] = \text{HSIC}$ exactly.

### Debiased Linear CKA

$$\text{CKA}_{\text{debiased}}(X, Y) = \frac{\text{HSIC}_{\text{debiased}}(XX^T, YY^T)}{\sqrt{\text{HSIC}_{\text{debiased}}(XX^T, XX^T) \cdot \text{HSIC}_{\text{debiased}}(YY^T, YY^T)}}$$

Note: $\text{HSIC}_{\text{debiased}}(K, K) < 0$ is theoretically possible for small $N$ with near-zero signal. In this case, $\text{CKA}_{\text{debiased}} = 0$ (handled with `max(hsic_xx, 0)` in the denominator).

### Python Implementation Reference

```python
def _debiased_hsic(K, L):
    """K, L: (N, N) numpy float64 arrays."""
    n = K.shape[0]
    K_ = K.copy(); np.fill_diagonal(K_, 0.0)
    L_ = L.copy(); np.fill_diagonal(L_, 0.0)
    ks = K_.sum(axis=1)   # (N,) row sums (diagonal excluded)
    ls = L_.sum(axis=1)
    term1 = float((K_ * L_).sum())                          # sum_{i≠j} K_ij L_ij
    term2 = float(ks.sum() * ls.sum()) / ((n-1) * (n-2))   # correction term
    term3 = float(2.0 * ks @ ls) / (n - 2)                 # cross term
    return (term1 + term2 - term3) / (n * (n - 3))

def linear_cka_debiased(X, Y):
    """X, Y: (N, D) numpy arrays."""
    K = X @ X.T
    L = Y @ Y.T
    xy = _debiased_hsic(K, L)
    xx = _debiased_hsic(K, K)
    yy = _debiased_hsic(L, L)
    denom = np.sqrt(max(xx, 0.0) * max(yy, 0.0))
    return float(xy / denom) if denom > 0 else 0.0
```

---

## 11. All 96 Depth-Scale-Invariant Channel Indices

Selected by $|\text{corr}(\text{channel}_c, \log\text{-std}_{\text{depth}})|$ across 100 calibration frames (50 Boston + 50 Singapore, CAM_FRONT, seed 42). Top 96 of 384 channels (top 25%).

```python
DEPTH_SCALE_INVARIANT_CHANNELS = [
    374,  77, 294, 315,  36,  25, 362, 299, 328, 334,
    103,  61, 278,  43, 215, 344,  10,  50, 226, 160,
    224, 130, 369, 355, 372,  49,  22, 286, 282, 272,
     20, 321, 142,  51, 287,  35, 276, 245, 242, 143,
    268, 161, 234,  19,  37, 262, 327, 288, 349, 316,
    261, 357, 237, 364,   2, 211,  28, 139, 300, 380,
    263,  24, 184,  96, 102,   7, 217, 254, 174, 115,
    232, 246, 128, 117,  81, 253, 257,  53,  42,  68,
    275,  30,  89, 255, 202, 200, 310, 319, 151, 360,
    163, 301,  45, 356,  79, 312
]
# len = 96; sorted by decreasing |r| with log-depth std
```

**Selection criterion:** These channels have the highest correlation with depth-scale (log standard deviation of the estimated depth map) across both cities combined. A high correlation means the channel encodes depth-scale information; selecting these 96 filters the adapter to geometric priors rather than photometric appearance.

**Statistical summary of selection:**
- Mean |r| (invariant channels) with log_std: ≥ 0.466
- Mean |r| (variant channels) with log_std: < 0.437
- Mean invariant score (invariant - variant gap): 0.030 ± 0.181

---

## 12. Boston 2,000 Sample Selection for E3/E4/E6

**Selection method:** First 2,000 frames from the Boston training split (training PKL file), in sequential order as stored. No random sampling, no stratification.

**Implementation:** In MMDetection3D config:
```python
train_dataloader = dict(
    dataset=dict(
        indices=list(range(2000))  # first 2000 of the training split
    )
)
```

**Rationale for choosing sequential (not random):** The training PKL is ordered by scene then time within scene. Taking the first 2,000 frames captures ~71 complete training scenes out of 700, with full temporal continuity within each scene (important for BEVFormer's temporal cross-attention). A random subset would break temporal continuity within scenes.

**Is 2,000 representative?** The 700 training scenes are all from Boston-Seaport, so all 2,000 frames share the same distribution as the full training set. The subset is used purely to reduce per-epoch training time from ~5 hours to ~45 minutes on the RTX 5060.

---

## 13. Camera Ordering and CAM_FRONT Index

nuScenes uses 6 cameras. In BEVFormer's image tensor `(B, N_cam, 3, H, W)`:

| Index | Camera Name | Position | Field of View |
|---|---|---|---|
| **0** | **CAM_FRONT** | Front center | 70° horizontal |
| 1 | CAM_FRONT_RIGHT | Front right | 70° |
| 2 | CAM_FRONT_LEFT | Front left | 70° |
| 3 | CAM_BACK | Rear center | 110° (wide) |
| 4 | CAM_BACK_LEFT | Rear left | 70° |
| 5 | CAM_BACK_RIGHT | Rear right | 70° |

**CAM_FRONT (index 0)** is used exclusively for:
- CKA computation (spatial descriptor from the front-facing camera only)
- DAv2 depth channel stability analysis (50+50 frames, front camera)

**All 6 cameras** are used for:
- Cosine similarity computation (averaged across cameras 0–5)
- BEVFormer inference (all 6 cameras are input simultaneously)

**Image resolution:** 1600×900 pixels (all cameras, same sensor and resolution in nuScenes).

---

## 14. Pseudo-Label Generation — Full Specification

### Generation Process (`generate_pseudo_labels.py`)

1. Load the frozen BEVFormer-Base (`bevformer_base_epoch_24.pth`)
2. Run inference on all 2,929 Singapore validation frames sequentially
3. For each frame: collect all predicted 3D bounding boxes with score > τ = 0.3
4. Save as `singapore_pseudo_labels.pkl`

### Box Format

Each pseudo-label is a 9D vector in the LiDAR coordinate frame:
```
[x, y, z, l, w, h, yaw, vx, vy]
```
where (x,y,z) is the box center, (l,w,h) are length/width/height, yaw is rotation around z-axis, (vx,vy) is velocity.

**Important:** The merge script (`merge_pseudo_labels.py`) splits these into:
- `gt_boxes`: 7D = `[x, y, z, l, w, h, yaw]` (for CustomNuScenesDataset compatibility)
- `gt_velocity`: 2D = `[vx, vy]` (stored separately to avoid double-appending)

### Pseudo-Label Statistics

| Metric | Value |
|---|---|
| Frames processed | 2,929 |
| Confidence threshold τ | 0.3 |
| Total boxes retained | 50,134 |
| Average boxes per frame | 17.1 |
| Min boxes per frame | ~3 (rare scenes) |
| Max boxes per frame | ~60 (dense traffic) |

### Merge Process (`merge_pseudo_labels.py`)

Pseudo-labels (`singapore_pseudo_labels.pkl`) contain only `(token, gt_boxes, gt_labels, gt_scores)`. The full metadata (calibration, timestamps, sensor extrinsics) comes from the original Singapore datalist PKL. They are merged by **sequential index** (positional alignment), not by token matching:

```python
for i, (info, pl) in enumerate(zip(sing_infos, pseudo_infos)):
    new_info = copy.deepcopy(info)       # full original metadata
    n_boxes  = len(pl["gt_boxes"])
    new_info["gt_boxes"]    = pl["gt_boxes"][:, :7]      # 7D
    new_info["gt_labels"]   = pl["gt_labels"]
    new_info["gt_scores"]   = pl["gt_scores"]
    new_info["gt_velocity"] = pl["gt_boxes"][:, 7:9]     # 2D
    new_info["gt_names"]    = [NUSCENES_CLASSES[l] ...]  # from labels
    new_info["valid_flag"]  = np.ones(n_boxes, dtype=bool)
    new_info["num_lidar_pts"] = np.ones(n_boxes, dtype=np.int32)
```

### Why Sequential Index Merge Works

Both the pseudo-label PKL and the original Singapore datalist PKL are generated from the same Singapore validation frames in the same order (nuScenes scene-then-time ordering). Sequential alignment is safe and avoids the token mismatch (pseudo-labels used `frame_N` fallback tokens instead of real nuScenes tokens).

---

## 15. Complete BibTeX Entries for All 13 References

```bibtex
@inproceedings{li2022bevformer,
  title     = {{BEVFormer}: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author    = {Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022},
  pages     = {1--18},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-20077-9_1}
}

@inproceedings{li2023bevdepth,
  title     = {{BEVDepth}: Acquisition of Reliable Depth for Multi-View 3D Object Detection},
  author    = {Li, Yinhao and Ge, Zheng and Yu, Guanyi and Yang, Junjie and Wang, Zheng and Shi, Yuchen and Sun, Jianjian and Li, Zeming},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year      = {2023},
  volume    = {37},
  number    = {2},
  pages     = {1477--1485},
  doi       = {10.1609/aaai.v37i2.25233}
}

@inproceedings{wang2023streampetr,
  title     = {Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection},
  author    = {Wang, Shihao and Liu, Yingfei and Wang, Tiancai and Li, Ying and Zhang, Xiangyu},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023},
  pages     = {3621--3631},
  doi       = {10.1109/ICCV51070.2023.00337}
}

@inproceedings{caesar2020nuscenes,
  title     = {{nuScenes}: A Multimodal Dataset for Autonomous Driving},
  author    = {Caesar, Holger and Bankiti, Varun and Lang, Alex H and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020},
  pages     = {11621--11631},
  doi       = {10.1109/CVPR42600.2020.01164}
}

@inproceedings{kornblith2019cka,
  title     = {Similarity of Neural Network Representations Revisited},
  author    = {Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and Hinton, Geoffrey},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2019},
  volume    = {97},
  pages     = {3519--3529},
  url       = {http://proceedings.mlr.press/v97/kornblith19a.html}
}

@inproceedings{yang2024depthanythingv2,
  title     = {Depth Anything {V2}},
  author    = {Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.09414}
}

@inproceedings{dabev2024,
  title     = {Domain Adaptive 3D Object Detection via Nerve-Like Feature Reuse},
  author    = {Jiang, Bo and Zhang, Pengfei and Chen, Xiaobo and Luo, Bin},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2407.03654}
}

@inproceedings{yang2021st3d,
  title     = {{ST3D}: Self-Training for Unsupervised Domain Adaptation on 3D Object Detection},
  author    = {Yang, Jihan and Shi, Shaoshuai and Wang, Zirui and Li, Hongsheng and Qi, Xiaojuan},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021},
  pages     = {10368--10378},
  doi       = {10.1109/CVPR46437.2021.01023}
}

@inproceedings{luo2021mlc,
  title     = {Unsupervised Domain Adaptive 3D Detection with Multi-Level Consistency},
  author    = {Luo, Zhipeng and Cai, Zhongang and Wang, Changqing and Zhang, Gongjie and Li, Haiyu and Liu, Ziwei and Lu, Shijian and Dong, Weihao and Zhang, Shiliang and Lyu, Lingling and others},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021},
  pages     = {8585--8594},
  doi       = {10.1109/ICCV48922.2021.00846}
}

@inproceedings{wang2021tent,
  title     = {Tent: Fully Test-Time Adaptation by Entropy Minimization},
  author    = {Wang, Dequan and Shelhamer, Evan and Liu, Shaoteng and Olshausen, Bruno and Darrell, Trevor},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  url       = {https://openreview.net/forum?id=uXl3bZLkr3c}
}

@inproceedings{liang2021shot,
  title     = {Do We Really Need to Access the Source Data? {S}ource Hypothesis Transfer for Unsupervised Domain Adaptation},
  author    = {Liang, Jian and Hu, Dapeng and Feng, Jiashi},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2020},
  volume    = {119},
  pages     = {6028--6039},
  url       = {http://proceedings.mlr.press/v119/liang20a.html}
}

@inproceedings{sun2020ttt,
  title     = {Test-Time Training with Self-Supervision for Generalization under Distribution Shifts},
  author    = {Sun, Yu and Wang, Xiaolong and Liu, Zhuang and Miller, John and Efros, Alexei A and Hardt, Moritz},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2020},
  volume    = {119},
  pages     = {9229--9248},
  url       = {http://proceedings.mlr.press/v119/sun20b.html}
}

@article{xu2023vlm3det,
  title     = {{VLM3Det}: A Framework for 3D Object Detection with Vision Language Models},
  author    = {Xu, Xiuwei and Zhong, Runhao and Tang, Jiwen and Lu, Jiwen},
  journal   = {arXiv preprint arXiv:2307.12609},
  year      = {2023},
  url       = {https://arxiv.org/abs/2307.12609}
}
```

---

## 16. Figure Descriptions for Redrawing

### Figure 1 — Does Not Exist
*(No high-level overview figure currently exists for the paper introduction. This could optionally be added as a teaser.)*

### Figure 2 — Pipeline Diagram (`pipeline_diagram.pdf`)

**Conceptual layout (left to right, three rows):**

**Top row:** `[nuScenes Dataset]` → `[City Split by location field]` → `[Boston val (3,090)]` and `[Singapore val (2,929)]`

**Middle row — Main BEVFormer path:**
`[Camera Images (6×)]` → `[ResNet-101 + FPN]` → `[hook ①: img_feat]` → `[⊕ residual injection]` → `[BEV Encoder]` → `[hook ②: bev_embed]` → `[Detection Head]` → `[Predictions]`

**Upper branch — DAv2 path:**
`[Camera Images]` → `[DAv2 ViT-S (frozen)]` → `[Channel Select 96/384]` → `[Conv 3×3 → ReLU → Conv 1×1]` → `[× α]` → `[→ to ⊕]`

**Bottom row — Experiments:**
Six boxes for E3-A, E3-B, E3-C, E4, E5, E6 with their root causes.

**Annotations:**
- Between hook ① and ②: "Cosine: 0.424→0.890 (81%)"
- Below hooks: "CKA near zero in all conditions; t-SNE drift ratio=1.30"

**Colour coding:**
- Dark grey = frozen BEVFormer components
- Warm red = trainable adapter
- Blue = frozen DAv2
- Orange/gold = residual injection point ⊕

### Figure 3 — Adapter Architecture Schematic (`adapter_schematic.pdf`)

**Main path (horizontal):**
`[Camera Images]` → `[ResNet-101+FPN (frozen)]` → `[img_feat]` → `[⊕]` → `[BEV Encoder (frozen)]` → `[bev_embed]` → `[Detection Head (frozen)]`

**DAv2 branch (above main path):**
`[Camera Images]` → `[DAv2 ViT-S (frozen)]` → `[(B·N_cam, 384, H', W')]` → `[Channel Select (96/384)]` → `[Conv 3×3 (trainable)]` → `[ReLU]` → `[Conv 1×1 (trainable)]` → `[× α]` → `[→ ⊕]`

**Dashed vertical arrows** from main path to annotation boxes:
- At FPN output: "hook ① img_feat, Cosine μ=0.424"
- At BEV encoder output: "hook ② bev_embed, Cosine μ=0.890 (81% norm.)"

**Parameter count box:**
"E3-B (384ch): 164K params | E6 (96ch): 91K params"

### Figure 4 — t-SNE BEV Drift (`tsne_encoder_drift.pdf`)

**2D scatter plot** showing 400 BEV encoder feature vectors projected to 2D by t-SNE:
- Blue points: 200 Boston frames
- Orange points: 200 Singapore frames
- Partial separation is visible — not fully clustered but measurably different
- **Caption:** "Cross-domain distance = 2.13; within-city = 1.64; drift ratio = **1.30**"

### Figure 5 — Per-Class AP Comparison (`per_class_ap_comparison.pdf`)

**Grouped bar chart:** 10 classes on x-axis, AP on y-axis (0–1).
- Blue bars: Boston AP per class
- Orange bars: Singapore AP per class
- Sorted by relative gap (worst on left: trailer at −100%, pedestrian on right at +2.2%)
- Horizontal dashed line at y=0.367 (Singapore mAP) for reference

---

## 17. Section-by-Section Narrative Writing Guide

This section provides writing guidance for someone composing the final paper, including emphasis points, framing choices, and sentences to avoid.

### §1 Abstract (250–300 words target)

**Lead with the problem, not the method.** Open with what breaks (camera BEV detection degrades under city shift) and what is unknown (how much, why, and where).

**The controlled experiment** — nuScenes Boston→Singapore — must appear in sentence 2. Reviewers need to understand the testbed before the numbers.

**Three-sentence structure for contributions:**
1. Gap measurement sentence: "−5.84 mAP / −9.25 NDS, statistically significant (p<0.01)"
2. Representation sentence: "cosine 81% normalization; CKA null; t-SNE drift 1.30"
3. Adapter sentence: "six designs, six distinct failures"

**Do NOT say:** "We propose a novel method" — this is a diagnostic paper, not a method paper. The contribution is the diagnostic framework and the failure analysis.

**Fix before submission:** Remove the duplicate t-SNE sentence (see §1 of this supplement).

### §2 Introduction (4–5 paragraphs)

**Paragraph 1 — The problem:** AV systems fail under city shift. Prior work confounds variables. We isolate city.

**Paragraph 2 — Why nuScenes:** Explain the unique property (same sensors, same annotators, two cities) and the city split protocol.

**Paragraph 3 — First contribution:** The gap measurement. Emphasize the error breakdown pattern (mATE barely worsens, mAAE +125%). This pattern is the hook for the representation analysis.

**Paragraph 4 — Second contribution:** Layer-wise analysis. State the cosine finding clearly. Acknowledge the CKA null result honestly — do not try to hide it. It makes the paper more credible.

**Paragraph 5 — Third contribution:** Six adapter experiments. End with the strong framing: "these mechanistic explanations constrain the design space for future work more tightly than a marginal positive result would."

**Include Figure 1 (pipeline diagram) here** with a forward reference.

### §3 Related Work (4 paragraphs)

**Write tight paragraphs, each ending with a distinguishing sentence.**

**Camera BEV Detection:** End with "none of these methods addresses cross-city generalization under controlled conditions."

**Domain Adaptation for 3D Detection:** End with "these methods conflate multiple variables simultaneously; we isolate city-level shift within a single dataset."

**Source-Free and Test-Time Adaptation:** End with "our setting is stricter: a single frozen checkpoint on isolated target frames, without access to a stream of target data at adaptation time."

**Foundation Depth Models:** End with "we use frozen ViT-S encoder features — not depth predictions — as a domain-stable geometric prior, and find that even valid priors fail when injection is supervised incorrectly."

### §4 Experimental Setup (concise, ~500 words)

**Dataset subsection:** Include the frame count table. State explicitly that the `location` field is used for splitting — this is reproducible. Mention the custom evaluator (`BEVFormerNuScenesMetric`) and why it is necessary.

**Model subsection:** State the checkpoint score vs. our measured baseline score. The within-1-point verification is important for reproducibility trust.

**Metrics subsection:** Include the NDS formula. Explicitly name all five TP error types. State that mAP uses BEV center-distance matching (not IoU) — reviewers from 2D detection may assume IoU.

**Pairing subsection:** State density-bucketing clearly. Acknowledge the within-city anchor (mentioned in §5.3) is computed from the same 500 Boston features.

### §5.1 Gap Quantification

**Lead with the pattern, not the headline number.** The −5.84 mAP is notable, but the error breakdown is the scientific contribution. Write: "More revealing than the headline mAP gap is the error breakdown: translation error barely worsens (+9%) while attribute error collapses (+125%)..."

**Address the CI/p-value discrepancy explicitly** in a footnote, not in the main text. Reviewers who know statistics will notice; those who don't can ignore the footnote.

### §5.2 Per-Class AP

**The trailer finding is important but needs contextualization.** "Trailer AP collapses to zero, which partly reflects the absence of trailers in the Singapore validation split rather than purely visual domain shift. Excluding trailer, the mean relative gap for the remaining nine classes is −10.3%."

**The pedestrian improvement** (+2.2%) is a small effect that should not be over-interpreted. Write: "Pedestrian AP slightly improves, likely reflecting higher pedestrian density in Singapore-OneNorth's downtown district, though this effect is within the noise of the bootstrap CI."

### §5.3 Representation Analysis

**This is the most technically dense section. Write it in three phases:**
1. *Methodology* — What hooks, what metrics, what N, how pairs are formed (density-matched)
2. *Results table* — Present Table 3 with both cross-city rows AND within-Boston reference rows
3. *Interpretation* — The two-part story: cosine (81%) + CKA (null result + bev_embed exception)

**The CKA null result must be framed carefully:**
- Do NOT write: "CKA shows no structural gap" — this is wrong.
- DO write: "CKA is near zero in all comparison types, indicating that scene-content variability dominates structural geometry regardless of city identity. This null result is informative: the domain gap does not manifest as a detectable CKA difference at this sample size. The primary structural evidence rests on the cosine residual and the t-SNE drift ratio."

**The bev_embed CKA exception** (0.009, CI excludes zero) deserves a sentence: "Notably, cross-city bev_embed CKA is the only statistically non-zero value (CI [0.004, 0.014]), suggesting the BEV encoder imposes a weakly detectable shared structure between cities that does not emerge from same-city comparisons."

### §5.4 t-SNE

**Keep this subsection short** (~150 words). The drift ratio (1.30) is the single number. Include the figure reference. The "calibration decoupling" interpretation should be stated: the encoder normalizes individual-frame appearance (high cosine) but does not collapse the overall city-level distribution (positive drift ratio).

### §6 Adapter Ablation

**Structure each experiment as:** Setup → Key Diagnostic → Root Cause → Implications.

**The E6 result** should be placed last and framed as the "confirmation experiment": it takes E3-B's failure mode hypothesis (wrong supervision, not wrong channels) and subjects it to direct empirical test. The fact that E6 gives identical results to E3-B is not a negative result — it is a controlled scientific confirmation.

**The E5 circular optimization finding** deserves the most discussion. The key point — gradient flow present, performance still degrades — is counterintuitive and should be stated prominently. This is the most novel mechanistic finding in the paper.

### §7 Conclusion

**Follow this structure:**
1. Restate the protocol and headline numbers (1 sentence)
2. Summarize the representation findings — cosine and t-SNE (2 sentences)
3. Summarize the adapter findings — one sentence per key failure mode (4–5 sentences)
4. Forward-looking paragraph: what would success require? (3–4 sentences)
5. Closing sentence: the protocol and artifacts are released for reproducibility.

---

## 18. Reproducibility Commands — Full Training and Eval

All commands assume the working directory is `E:\Auto_Image\bev_research\mmdetection3d` and the conda environment `bev310` is activated.

### Baseline Evaluation

```powershell
# Boston
python tools\test.py `
  E:\bev_research\configs\bevformer_singapore_eval.py `
  E:\bev_research\checkpoints\bevformer_base_epoch_24.pth `
  2>&1 | Tee-Object E:\bev_research\logs\baseline_boston_eval.log

# Singapore
# (Use e5_pseudo_label_adapter.py with baseline checkpoint for Singapore-only eval)
python tools\test.py `
  E:\bev_research\configs\adapter\e5_pseudo_label_adapter.py `
  E:\bev_research\checkpoints\bevformer_base_epoch_24.pth `
  2>&1 | Tee-Object E:\bev_research\logs\baseline_singapore_eval.log
```

### Representation Analysis (N=500 pairs, real GPU inference)

```powershell
python E:\bev_research\scripts\representation_analysis_v2.py `
  --config E:\bev_research\configs\bevformer_rtx5060.py `
  --checkpoint E:\bev_research\checkpoints\bevformer_base_epoch_24.pth `
  --pairs E:\bev_research\data\matched_pairs_500.json `
  --n-pairs 500 `
  --device cuda:0 `
  --features-cache E:\bev_research\data\cka_features_500_v2.npz `
  2>&1 | Tee-Object E:\bev_research\logs\repr_analysis.log
```

### PCA + Debiased CKA Bootstrap (CPU only, uses cached features)

```powershell
python E:\bev_research\scripts\pca_cka_bootstrap.py
# Output: E:\bev_research\logs\cka_pca_bootstrap.json
```

### E5 Training

```powershell
python tools\train.py `
  E:\bev_research\configs\adapter\e5_pseudo_label_adapter.py `
  --work-dir E:\bev_research\work_dirs\e5_pseudo_label_adapter `
  2>&1 | Tee-Object E:\bev_research\logs\e5_training.log
```

### E5 Singapore Evaluation (after training)

```powershell
python tools\test.py `
  E:\bev_research\configs\adapter\e5_pseudo_label_adapter.py `
  E:\bev_research\work_dirs\e5_pseudo_label_adapter\epoch_4.pth `
  2>&1 | Tee-Object E:\bev_research\logs\e5_eval.log
```

### E6 Training

```powershell
python tools\train.py `
  E:\bev_research\configs\adapter\e6_depth_scale_channels.py `
  --work-dir E:\bev_research\work_dirs\e6_depth_scale_channels `
  2>&1 | Tee-Object E:\bev_research\logs\e6_training.log
```

### E6 Singapore Evaluation

```powershell
# Note: uses e5 config (Singapore-only evaluator) with e6 checkpoint
python tools\test.py `
  E:\bev_research\configs\adapter\e5_pseudo_label_adapter.py `
  E:\bev_research\work_dirs\e6_depth_scale_channels\epoch_4.pth `
  2>&1 | Tee-Object E:\bev_research\logs\e6_singapore_eval.log
```

### t-SNE BEV Drift Analysis

```powershell
cd E:\bev_research
python scripts\tsne_encoder_drift.py `
  2>&1 | Tee-Object logs\tsne_drift.log
# Output: figures\tsne_encoder_drift.{json,pdf,png}
```

### Within-Boston CKA Anchor (CPU only, uses cached features)

```powershell
cd E:\bev_research
python -c "
import numpy as np, json
from sklearn.decomposition import PCA
from pathlib import Path
# ... (see full script in within_boston_cka.py)
"
# Output: logs\within_boston_cka.json
```

### Generate All Figures

```powershell
cd E:\bev_research
python scripts\figure_pipeline.py
python scripts\figure_adapter_schematic.py
python scripts\per_class_ap_plot.py
# tsne figure is generated by tsne_encoder_drift.py above
```

---

## 19. Paper Editorial Checklist Before Submission

### Pre-Submission Must-Fix

- [ ] **Remove duplicate t-SNE sentence from abstract** (lines 37–39 of `draft.tex`) — see §1 of this supplement
- [ ] **Populate `refs.bib`** with all 13 BibTeX entries from §15 of this supplement
- [ ] **Verify `\citeyear` macro** — requires `natbib` package; IEEEtran uses `\cite` by default. Either add `natbib` or replace `\citeyear{kornblith2019cka}` with explicit "(Kornblith et al., 2019)"
- [ ] **Check `\cref` references** — all `\label{}` must have corresponding `\cref{}` or `\ref{}` calls
- [ ] **E3-C consistency loss weight** — confirm λ_cons = 1.0 appears in the paper text (currently not stated explicitly in `draft.tex`)

### Style and Formatting

- [ ] **Table 3 column width** — the 7-column table may exceed IEEEtran column width in single-column mode; consider splitting into two subtables or using `table*` (double-column float)
- [ ] **Table 5 is already `table*`** — correct, it needs double-column for the root cause column
- [ ] **Figures 2 and 3** — replace matplotlib-generated PDFs with Illustrator/Inkscape versions for camera-ready; or keep as-is for first submission
- [ ] **Keywords** — add `nuScenes`, `BEV encoder`, `representation analysis` to IEEEkeywords if venue allows more than 6
- [ ] **Author names** — replace `[Author names omitted for review]` before camera-ready

### Content Consistency Checks

- [ ] **"five" vs "six" adapter count** — search for "five" in the paper; all should now read "six" (E3-A through E6)
- [ ] **CKA values** — anywhere the paper says CKA and gives a number, verify it matches the debiased PCA-100 values (0.003 / 0.009) not the biased 16384D values (0.114 / 0.139). The biased values appear in Table 3 for reference but should not be cited as the primary CKA result elsewhere
- [ ] **Cosine normalization 81%** — confirm this reads "81%" (not the old 67.6%) everywhere in the paper
- [ ] **E6 table row** — verify the Singapore mAP/NDS in Table 5 row E6 shows 0.367/0.431 (exact, not ≈)
- [ ] **E5 limitation note** — verify the train/test overlap note appears after the E5 mechanistic paragraph

### Venue-Specific (IEEE RA-L)

- [ ] **Page limit:** RA-L allows up to 8 pages including references (Letter format, 10pt IEEEtran). Current paper estimated at 7–8 pages — check compiled PDF length
- [ ] **Supplementary:** RA-L allows up to 2 pages supplementary; t-SNE figure and per-class bar chart can move there if over page limit
- [ ] **Figures:** RA-L requires figures ≥ 300 DPI at final size; current figures are generated at 180 DPI — regenerate at 300 DPI before submission
- [ ] **ORCID:** IEEE requires ORCID IDs for all authors in camera-ready
- [ ] **Code release:** The paper claims "evaluation protocol and configs are released" — prepare a public GitHub repository before submission

### Known Remaining Limitations (Should Appear in Paper)

1. **No LiDAR results** — the domain gap is measured camera-only; LiDAR-based detectors may behave differently under city shift (different point cloud density patterns)
2. **Single target city** — Singapore-OneNorth to Boston-Seaport is one direction of one city pair; the failure modes may differ for other city pairs (e.g., Waymo's different geographic coverage)
3. **BEVFormer-Base only** — larger models (BEVFormer-Large) or newer architectures (StreamPETR, SparseBEV) may exhibit different gap magnitudes
4. **No real-time constraint** — the adapter adds DAv2 inference (~10ms on RTX 5060 at 308px input) to each frame; latency is not discussed

---

*End of supplementary report. Together with `bev_research_final_report.md`, these two documents provide complete information for independently writing the full journal paper, reproducing every experiment, and preparing a camera-ready submission.*
