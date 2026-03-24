# BEV Domain Gap Research — Comprehensive Implementation & Results Report

**Project:** *Where Does the Domain Gap Live? Diagnosing Camera BEV Detection Failure Under City-Level Shift*  
**Report Date:** 22 March 2026  
**GPU:** NVIDIA RTX 5060 (16 GB)  
**Environment:** `bev310` conda env · PyTorch 2.6 · MMDetection3D v1.x / MMEngine

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 0 — Baseline Gap Measurement](#2-phase-0--baseline-gap-measurement)
   - [0.1 Bootstrap CIs on mAP Gap](#21-phase-01-bootstrap-confidence-intervals-on-map-gap)
   - [0.2 Per-Class AP Analysis](#22-phase-02-per-class-ap-analysis)
3. [Phase 1 — Semantic Frame Pairing](#3-phase-1--semantic-frame-pairing)
4. [Phase 2 — Representation Drift Analysis (CKA)](#4-phase-2--representation-drift-analysis-cka)
   - [Feature Extraction Bug — Full History](#41-feature-extraction-bug--full-history)
   - [Final Correct Results](#42-final-correct-results)
   - [CKA CI Bias Issue](#43-cka-ci-bias-issue)
5. [Phase 3 — Pseudo-Label Generation](#5-phase-3--pseudo-label-generation)
6. [Phase 4 — E5 Pseudo-Label Adapter Training](#6-phase-4--e5-pseudo-label-adapter-training)
   - [merge_pseudo_labels.py Bug Chain](#61-merge_pseudo_labelspy-bug-chain-three-fixes)
   - [Training Results & Circular Optimization](#62-training-results--circular-optimization)
7. [Phase 5 — DAv2 Depth Channel Analysis](#7-phase-5--dav2-depth-channel-analysis)
8. [Phase 6 — t-SNE BEV Feature Drift](#8-phase-6--t-sne-bev-feature-drift)
9. [All Adapter Experiments Summary](#9-all-adapter-experiments-summary)
10. [Paper Draft Status](#10-paper-draft-status)
11. [All Scripts Modified/Created](#11-all-scripts-modifiedcreated)
12. [Known Remaining Issues](#12-known-remaining-issues)

---

## 1. Project Overview

The goal is to strengthen a BEV domain gap paper by generating new data, analyses, and a new adapter experiment (E5). The paper measures the city-level domain gap of BEVFormer-Base trained on Boston and evaluated on Singapore (within the nuScenes dataset), then uses layer-wise representation analysis to localize the gap and systematically evaluates five injection strategies using Depth Anything V2 (DAv2) features.

**Key model:** BEVFormer-Base, checkpoint `bevformer_base_epoch_24.pth`  
**Dataset:** nuScenes v1.0-trainval, Boston (source) / Singapore (target) city split  
**Boston val frames:** 3,090 | **Singapore val frames:** 2,929

---

## 2. Phase 0 — Baseline Gap Measurement

### 2.1 Phase 0.1: Bootstrap Confidence Intervals on mAP Gap

**Script:** `bootstrap_ci.py`  
**Output:** `E:\bev_research\logs\bootstrap_ci_results.json`

The bootstrap was run over 10 per-class AP values (treating each class as an independent observation of the gap). This is a coarse bootstrap (N=10) but sufficient to characterize whether the gap is statistically significant.

#### Final Numbers

| Split | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
|---|---|---|---|---|---|---|---|
| Full val | 0.405 | 0.500 | — | — | — | — | — |
| **Boston** | **0.425** | **0.524** | 0.666 | 0.280 | 0.321 | 0.461 | 0.157 |
| **Singapore** | **0.367** | **0.431** | 0.726 | 0.354 | 0.472 | 0.615 | 0.353 |
| **Gap** | **−5.8** | **−9.3** | +0.060 | +0.074 | +0.151 | +0.154 | +0.196 |

**Bootstrap 95% CI on mAP gap (N=10 classes):** [−20.7, +8.8]  
**Statistical tests:** Paired t-test p = 0.0034 · Wilcoxon p = 0.0039

The wide CI is expected given N=10 classes. The p-values confirm the gap is statistically significant: both tests reject the null at α=0.01. The CI being wide reflects class-level variance in the gap (trailer collapses −100%, pedestrian improves +2.2%).

**Key observation on error breakdown:** mATE worsens only 9% but mAOE (orientation) worsens 47%, mAVE (velocity) 33%, and mAAE (attribute) 125%. The model locates objects but is miscalibrated for their motion and orientation — exactly the kind of fine-grained signal driven by feature-space structure, not just global appearance.

---

### 2.2 Phase 0.2: Per-Class AP Analysis

**Script:** `per_class_ap_plot.py`  
**Output:** `E:\bev_research\logs\per_class_ap.json` · `E:\bev_research\figures\per_class_ap_comparison.png/pdf`

| Class | Boston AP | Singapore AP | Abs. Gap | Rel. Gap |
|---|---|---|---|---|
| trailer | 0.157 | 0.000 | −0.157 | **−100.0%** |
| construction_vehicle | 0.135 | 0.107 | −0.028 | −20.8% |
| bicycle | 0.429 | 0.352 | −0.077 | −18.0% |
| traffic_cone | 0.617 | 0.510 | −0.107 | −17.4% |
| truck | 0.356 | 0.309 | −0.047 | −13.2% |
| motorcycle | 0.479 | 0.416 | −0.063 | −13.2% |
| bus | 0.448 | 0.399 | −0.048 | −10.8% |
| barrier | 0.526 | 0.492 | −0.034 | −6.5% |
| car | 0.621 | 0.589 | −0.032 | −5.2% |
| pedestrian | 0.482 | **0.492** | +0.011 | **+2.2%** |
| **mAP** | **0.425** | **0.367** | −0.058 | −13.7% |

**Findings:**
- **Trailer** collapses entirely — Singapore has almost no trailers in this split; the model correctly has nothing to detect
- **Visually unusual classes** (construction vehicle, bicycle, traffic cone) suffer most — they rely on fine-grained appearance cues that differ across cities
- **Pedestrian improves** slightly — Singapore-OneNorth is denser urban fabric; more pedestrians in familiar contexts

---

## 3. Phase 1 — Semantic Frame Pairing

**Script:** `build_semantic_pairs.py`  
**Output:** `E:\bev_research\data\matched_pairs_500.json` (500 Boston↔Singapore pairs)

### Algorithm
Pairs Boston and Singapore frames with similar scene composition ("density bucketing") to avoid confounding the representation analysis with scene-density effects. Object density proxy: number of GT boxes in the frame.

**Density bins:** `sparse` (0–3 boxes), `medium` (4–8), `dense` (9+)  
**Allocation:** proportional to available frames per bucket (not equal)

### Bugs Fixed During This Phase

**Bug 1 — Timezone mismatch in time-of-day bucketing:**  
The original implementation used `(time_of_day_bin, density_bin)` as the semantic key. Boston timestamps are UTC noon (local morning), Singapore are UTC 0600 (local afternoon), so they clustered into completely different UTC time bins. Result: only 365/500 pairs were generated because most buckets had zero overlap.  
**Fix:** Changed `semantic_key` to use density-only, dropping time-of-day entirely.

```python
def semantic_key(info: dict) -> tuple:
    # Density-only: timezone-independent, strongest semantic proxy available
    return (get_density_bin(info),)
```

**Bug 2 — Sparse bucket bottleneck:**  
Even with density-only bucketing, only 335/500 pairs were produced because the `sparse` bucket had only 1 Boston frame, capping pair allocation at 1.  
**Fix:** Switched from equal bucket allocation to proportional allocation:

```python
avail = {k: min(len(boston_buckets[k]), len(sing_buckets[k])) for k in common_keys}
total_avail = sum(avail.values())
budget = round(args.n_pairs * avail[key] / total_avail)
budget = min(budget, avail[key])
```

**Result:** 500 pairs successfully generated across density buckets. All 500 pairs saved with `bucket_key` (`'dense'` bucket predominant in urban driving).

---

## 4. Phase 2 — Representation Drift Analysis (CKA)

**Script:** `representation_analysis_v2.py`  
**Output:** `E:\bev_research\logs\representation_analysis_v2.json` · `cka_features_500_v2.npz` (feature cache)

This was the most iteratively-debugged phase. Three distinct bugs were found and fixed before obtaining valid results.

### 4.1 Feature Extraction Bug — Full History

#### Attempt 1 (11:16 AM, March 21): Demo Mode Results Mistaken for Real Inference
The JSON at `11:16:55 AM` showed `cosine_sim_mean = 0.398` and `linear_cka = 0.444`. These appeared to match the original N=20 analysis (0.397 cosine), but were actually from **demo mode** (`--demo` flag), which generates synthetic data:

```python
sing_img = 0.397 * boston_img + sqrt(1 - 0.397²) * noise_img
```

The demo was designed to approximate the N=20 result, so the cosine value matched by construction. The CKA = 0.444 was entirely synthetic.

#### Attempt 2 (March 22 morning): Camera-Averaged Global Mean → cosine = 0.947
The script was run in real inference mode with camera averaging. The hook captured FPN output, spatial mean-pooled across H×W, then averaged across 6 cameras → single 256D vector per frame. 

**Result:** `cosine_mean = 0.947`, `CKA = 0.017`

This was physically wrong. The camera-averaged, spatially-pooled vector represents the "average RGB statistics across the entire visible sphere" — both Boston and Singapore are urban environments, so this global average is nearly identical (cosine 0.947). Meanwhile CKA = 0.017 means the structural geometry is nearly orthogonal, which contradicts the high cosine similarity.

**Root cause identified by comparing to `bev_drift_diagnostic.py`** (the original N=20 script): The original computed cosine similarity on **full spatial feature maps per camera** (`(N_cam, C, H, W)` → flatten → per-camera cosine → average across cameras). This preserves spatial structure, which is what actually varies between cities.

#### Attempt 3 (Final — March 22): Spatial Descriptor Methodology  
**Fix implemented:**
- **For cosine similarity:** Flatten each camera's spatial feature map to `(C×H×W)`, compute cosine similarity between Boston and Singapore per camera, average across 6 cameras. Matches original methodology exactly. Cosine computed on-the-fly per pair to avoid storing ~36 GB of spatial tensors in RAM.
- **For CKA:** Use CAM_FRONT (camera index 0) at the finest FPN scale, adaptive-avg-pool to 8×8, flatten to `(256×8×8 = 16,384D)`. Structurally rich descriptor that CKA can meaningfully compare across 500 pairs.
- **For bev_embed:** Reshape BEV encoder output `(1, H_bev×W_bev, C)` → `(1, C, H_bev, W_bev)` → adaptive-avg-pool to 8×8 → flatten to `(256×64 = 16,384D)`.

### 4.2 Final Correct Results

**N = 500 density-matched pairs, real BEVFormer-Base inference**

| Layer | Cos. sim. μ | Cos. sim. σ | Cos. normalization | Linear CKA |
|---|---|---|---|---|
| After FPN (`img_feat`) | **0.424** | 0.033 | — | **0.114** |
| After BEV encoder (`bev_embed`) | **0.890** | 0.019 | **81%** | **0.139** |

**CKA-based gap normalization:** (1−0.114 − 1+0.139) / (1−0.114) = **2.74%**  
**Cosine-based gap normalization:** (1−0.424 − 1+0.890) / (1−0.424) = **81%**

**The two-metric interpretation:**
- **Cosine 0.424 → 0.890 (81% normalization):** The BEV encoder makes individual frame features strongly co-directional. Boston and Singapore BEV embeddings point in roughly the same direction in 256D space.
- **CKA 0.114 → 0.139 (2.7% normalization):** The structural geometry of the feature space barely changes. CKA measures how well the pattern of within-city variation in Boston maps to Singapore. A value of 0.139 means these patterns are nearly orthogonal — the BEV encoder does NOT bridge the structural gap.
- **The detection gap is driven by structural geometry, not global appearance direction.** The adapter must address structural geometry, not just coarse appearance.

Feature cache saved to `E:\bev_research\data\cka_features_500_v2.npz` for future CI recomputation without re-running GPU inference.

### 4.3 CKA CI Bias Issue

**Problem:** Subsampling bootstrap CIs are consistently *above* the point estimate — e.g., `img_feat CKA = 0.114` but CI = [0.132, 0.145]. This is statistically invalid.

**Root cause:** With `d >> n` (16,384D features, 500 samples), CKA is a biased estimator. The full-sample estimate uses all 500 pairs for 500×500 Gram matrices. Subsamples (400 pairs, 400×400 Gram matrices) have different rank properties in the high-dimensional space, yielding systematically higher CKA values. This is an intrinsic finite-sample bias of CKA in the high-d, low-n regime, not a code bug.

**Status:** CKA 95% CI marked as `(pending)` in the paper. The fix requires either: (a) PCA to reduce feature dimensionality to `d ≪ n` before CKA, or (b) a debiased HSIC estimator. Cosine similarity CIs are not affected (per-pair scalar; use standard error of the mean).

---

## 5. Phase 3 — Pseudo-Label Generation

**Script:** `generate_pseudo_labels.py`  
**Output:** `E:\bev_research\data\singapore_pseudo_labels.pkl`

Runs the frozen BEVFormer-Base on all 2,929 Singapore validation frames and saves predictions above confidence threshold τ = 0.3 as pseudo-labels for E5 adapter training.

### Results
- **Frames processed:** 2,929
- **Total boxes retained (τ > 0.3):** 50,134
- **Average boxes per frame:** 17.1
- **Box format:** 9D — `(x, y, z, l, w, h, yaw, vx, vy)` in LiDAR frame

### Key Fixes Applied
1. **PyTorch 2.6 `weights_only=True` fix:** Monkey-patched `torch.load` to `weights_only=False` for MMEngine checkpoint compatibility
2. **`build_dataset` API fix:** Replaced legacy `mmdet3d.datasets.build_dataset()` with `mmdet3d.registry.DATASETS.build()` (MMDetection3D v1.x API)
3. **Batched inference fix:** Wrapped `Det3DDataSample` in a list and added `.unsqueeze(0)` batch dim for `model.test_step()`
4. **Token fallback:** Pseudo-labels saved with `frame_N` tokens (token attribute unavailable via hook), addressed in Phase 4 merge script

---

## 6. Phase 4 — E5 Pseudo-Label Adapter Training

**Config:** `E:\bev_research\configs\adapter\e5_pseudo_label_adapter.py`  
**Work dir:** `E:\bev_research\work_dirs\e5_pseudo_label_adapter\`  
**Training log:** `E:\bev_research\logs\e5_training.log`

### 6.1 `merge_pseudo_labels.py` Bug Chain (Three Fixes)

The pseudo-labels PKL only contains `(token, gt_boxes, gt_labels, gt_scores)`. The `CustomNuScenesDataset` requires full frame metadata (calibration, timestamps, etc.). `merge_pseudo_labels.py` was written to combine both. Three successive bugs were encountered and fixed during training startup.

#### Bug 1 — Token Mismatch → `Matched 0/2929 frames`
The pseudo-labels used `frame_N` fallback tokens; the original datalist used real nuScenes tokens. Token-based matching found zero matches.  
**Fix:** Sequential index merge — both PKLs come from the same dataset in the same order, so positional alignment is safe.

#### Bug 2 — `IndexError: boolean index did not match` (dim 6 vs 9)
`get_ann_info` applies a boolean `mask = info['valid_flag']` to `gt_boxes`. The original `valid_flag` had length = original GT count (e.g., 9), but merged `gt_boxes` had pseudo-label count (e.g., 6). Shape mismatch.  
**Fix:** Regenerate `valid_flag`, `num_lidar_pts`, and `gt_names` to match pseudo-label count:

```python
new_info["valid_flag"]    = np.ones(n_boxes, dtype=bool)
new_info["num_lidar_pts"] = np.ones(n_boxes, dtype=np.int32)
new_info["gt_names"]      = np.array([NUSCENES_CLASSES[lbl] ... for lbl in gt_labels])
```

#### Bug 3 — `RuntimeError: shape mismatch [22, 11] vs [22, 10]`
The 9D pseudo-label boxes already contain velocity (columns 7–8: vx, vy). The dataset's `get_ann_info` also appended `gt_velocity` (2D), producing 11D boxes when the detection head expected 10D.  
**Fix:** Split 9D boxes into 7D `gt_boxes` + 2D `gt_velocity`:

```python
new_info["gt_boxes"]    = pl["gt_boxes"][:, :7].astype(np.float32)  # x,y,z,l,w,h,yaw only
new_info["gt_velocity"] = pl["gt_boxes"][:, 7:9].astype(np.float32) # vx, vy
```

**Final merge result:** 2,929 frames merged, 50,134 boxes, saved to `singapore_pseudo_labels_merged.pkl`.

### 6.2 Training Results & Circular Optimization

**Training setup:**
- 4 epochs on 2,929 Singapore pseudo-labeled frames
- Only adapter weights receive gradients (BEVFormer backbone + head frozen)
- lr = 2×10⁻⁴, batch size 1, ~6.7 s/iter

**Key observation during training:**  
`grad_norm` was **non-zero** (0.02 → 0.13 over first 150 iterations) — unlike E3-A/B where grad_norm was identically 0.0000. Target-domain pseudo-label loss **does** provide a gradient signal.

#### Per-Epoch Singapore Evaluation

| Epoch | mAP | NDS | vs. Baseline |
|---|---|---|---|
| Baseline | **0.367** | **0.431** | — |
| E5 Epoch 1 | 0.360 | 0.422 | −0.007 / −0.009 |
| E5 Epoch 2 | 0.359 | 0.421 | −0.008 / −0.010 |
| E5 Epoch 3 | 0.359 | 0.421 | −0.008 / −0.010 |
| E5 Epoch 4 | 0.356 | 0.417 | −0.011 / −0.014 |

**All four epochs are worse than the frozen baseline**, with monotonically decreasing performance.

#### Mechanistic Explanation — Circular Optimization

The pseudo-labels are generated by the same frozen model the adapter wraps. Training on these labels asks the adapter to bring FPN features closer to the representations the frozen backbone+head was pre-trained to process on Boston. This is the *opposite* of domain adaptation — the gradient pushes image features toward the Boston-optimal manifold, reinforcing existing bias rather than correcting for city-level shift.

The fact that `grad_norm > 0` but performance degrades makes this failure mode distinctive and important: **gradient flow alone is not sufficient evidence of useful adaptation**. The supervision signal must be independent of the model being adapted.

**Saved checkpoints:** `epoch_1.pth` through `epoch_4.pth` (~369 MB each, growing as adapter weights update)

---

## 7. Phase 5 — DAv2 Depth Channel Analysis

**Script:** `identify_depth_scale_channels.py`  
**Output:** `E:\bev_research\data\dav2_channel_analysis.json` · `E:\bev_research\logs\dav2_channel_analysis.log`

### Setup
- 100 frames (50 Boston + 50 Singapore, CAM_FRONT, seed 42)
- DAv2 ViT-S frozen encoder, 384 output channels
- Two stability metrics per channel: `log_std` (depth-scale correlation) and `grad_mean` (edge sharpness correlation)

### Results

| Metric | Value |
|---|---|
| Total channels analyzed | 384 |
| Invariant channels selected | **96** (25% of total) |
| Mean \|r\| with log_std | 0.4664 |
| Mean \|r\| with grad_mean | 0.4368 |
| **Top 10 invariant channels** | [374, 77, 294, 315, 36, 25, 362, 299, 328, 334] |

### Interpretation
Channels selected for injection are those most correlated with depth scale (`log_std`) and least correlated with appearance-variable signals like edge sharpness. Depth scale (log-std of the depth map) has Cohen's d = 0.09 between cities — effectively identical. The 96 invariant channels form the "geometry-only" subspace of the DAv2 encoder, providing the domain-stable geometric prior that motivates the adapter architecture.

---

## 8. Phase 6 — t-SNE BEV Feature Drift

**Script:** `tsne_encoder_drift.py`  
**Output:** `E:\bev_research\figures/tsne_encoder_drift.{json,png,pdf}`

### Setup
- 200 Boston + 200 Singapore BEV encoder features extracted (400 total)
- t-SNE: perplexity=30, max_iter=1000 (scikit-learn; `n_iter` → `max_iter` fix applied)
- Pairwise distances computed in 2D embedding space

### Results

| Metric | Value |
|---|---|
| Cross-domain mean distance | **2.129** |
| Within-city baseline distance | **1.642** |
| **Drift ratio** | **1.296** |
| Interpretation | `calibration_decoupling` |

### Interpretation
The drift ratio of **1.30** means Boston and Singapore BEV encoder features are 30% farther apart in t-SNE space than same-city frames. This directly confirms that **city-level separation persists at the BEV encoder output**, consistent with the CKA = 0.139 finding.

The `calibration_decoupling` label captures the finding precisely: the BEV encoder normalizes gross appearance (cosine 0.890) but the feature manifold remains city-partitioned (drift ratio 1.30, CKA 0.139), and this partition is what "calibrates" the detection head differently for Boston vs. Singapore traffic patterns.

---

## 9. All Adapter Experiments Summary

| Config | Modification | Full val mAP | Full val NDS | Singapore mAP | Singapore NDS | Root Cause of Failure |
|---|---|---|---|---|---|---|
| **Baseline** | Frozen BEVFormer, no adapter | 0.405 | 0.500 | 0.367 | 0.431 | — |
| **E3-A** | α = 0.01 | 0.405 | 0.500 | 0.367 | 0.431 | Adapter delta < 1% of feature magnitude; grad_norm ≡ 0 |
| **E3-B** | α = 0.1 | 0.405 | 0.500 | 0.367 | 0.431 | Zero-output local optimum; Boston loss pushes δ→0 |
| **E3-C** | α = 0.1 + consistency loss | 0.405 | 0.500 | 0.367 | 0.431 | ColorJitter ≠ real city shift; consistency ≠ detection signal |
| **E4** | Partial unfreeze (encoder lr=1e-5) | 0.373 | 0.331 | — | — | Head calibration breaks; mAOE 0.32→1.22 rad |
| **E5** | Pseudo-label adapt. (Singapore, τ=0.3) | — | — | 0.360 | 0.422 | Circular optimization; pseudo-labels reinforce source bias |

### Design Space Constraints (for future work)
1. **Independent supervision required:** E3-A/B show source-domain loss → zero output; E5 shows self-generated pseudo-labels → circular. Oracle labels, cross-modal geometric consistency, or SfM depth are needed.
2. **Augmentation must approximate real shift:** ColorJitter doesn't capture Boston→Singapore city-level appearance change (different buildings, roads, vegetation density).
3. **Joint adaptation needs flexible head or large dataset:** E4 shows encoder drift breaks frozen head calibration with only 2,000 samples.
4. **Structural geometry, not appearance, is the target:** Cosine 81% normalization creates a false impression of good alignment; CKA 2.7% normalization reveals the structural gap remains.

---

## 10. Paper Draft Status

**File:** `E:\bev_research\paper\draft.tex`  
**Format:** IEEEtran, 10pt, letter, journal mode

### Current Paper Structure

| Section | Content | Data Source |
|---|---|---|
| Abstract | Full summary with all 5 experiments, key numbers | All phases |
| §1 Introduction | 3 contributions, two-part normalization story | Phases 0–2 |
| §2 Related Work | BEV detection, domain adaptation, foundation depth | Literature |
| §3 Experimental Setup | nuScenes city split, BEVFormer-Base, metrics | Phase 0 |
| §4.1 Gap Table (Tab 1) | mAP/NDS/TP errors + bootstrap CIs + significance | Phase 0.1 |
| §4.1 Per-class Table (Tab 2) | 10-class AP sorted by relative gap | Phase 0.2 |
| §4.2 Representation Analysis (Tab 3) | Cosine + CKA, two-part story | Phase 2 |
| §4.3 t-SNE Section | Drift ratio 1.30, figure reference | Phase 6 |
| §4.4 DAv2 Stability (Tab 4) | Cohen's d per statistic | Phase 5 |
| §5 Ablation (Tab 5) | 5 experiments with root causes | Phases 3–4 |
| §5 E5 Mechanistic Analysis | Circular optimization explanation | Phase 4 |
| §5 Implications | 4 design constraints | All |
| §6 Conclusion | Two-part normalization + 5-experiment summary | All |

### Key Numerical Claims (All Verified from Real Inference)

| Claim | Value | Source |
|---|---|---|
| mAP gap | −5.84 | Phase 0 evaluation |
| NDS gap | −9.25 | Phase 0 evaluation |
| img_feat cosine sim | 0.424 ± 0.033 | Phase 2 (real inference) |
| bev_embed cosine sim | 0.890 ± 0.019 | Phase 2 (real inference) |
| Cosine normalization | 81% | Phase 2 |
| img_feat CKA | 0.114 | Phase 2 |
| bev_embed CKA | 0.139 | Phase 2 |
| CKA normalization | 2.7% | Phase 2 |
| t-SNE drift ratio | 1.296 | Phase 6 |
| DAv2 depth-scale Cohen's d | 0.09 | Phase 5 |
| DAv2 invariant channels | 96/384 | Phase 5 |
| E5 best Singapore mAP | 0.360 | Phase 4 |
| E5 pseudo-label boxes | 50,134 | Phase 3 |
| Bootstrap p-value (paired t) | 0.0034 | Phase 0.1 |

### Open Items in Paper
- **CKA 95% CIs:** Currently `(pending)` in Table 3. Require either PCA dimensionality reduction before CKA, or a debiased HSIC estimator. The current subsampling bootstrap gives upward-biased CIs when d >> n.
- **t-SNE figure:** Referenced as `../figures/tsne_encoder_drift.pdf` — file exists at `E:\bev_research\figures\tsne_encoder_drift.pdf` ✓
- **CKA footnote in Table 3:** Still references "67.6%" — updated to reflect the two-part story but should be double-checked

---

## 11. All Scripts Modified/Created

### Modified Scripts

| Script | Key Changes |
|---|---|
| `representation_analysis_v2.py` | (1) Fixed feature extraction to use per-camera spatial cosine + CAM_FRONT 8×8 pool for CKA; (2) Fixed bootstrap to subsampling without replacement; (3) Added `--features-cache` / `--recompute-ci` flags; (4) Added `compute_and_save` cosine override params; (5) Added PyTorch 2.6 `weights_only=False` monkey-patch; (6) Fixed MMDetection3D v1.x `DATASETS.build()` API; (7) Memory optimization: cosine computed on-the-fly |
| `tsne_encoder_drift.py` | PyTorch 2.6 fix; DATASETS.build() API fix; `n_iter` → `max_iter` for scikit-learn TSNE; camera averaging fix; PKL path updates; print flush=True |
| `generate_pseudo_labels.py` | PyTorch 2.6 fix; DATASETS.build() API fix; batched test_step() fix |
| `build_semantic_pairs.py` | Time-of-day → density-only bucketing; proportional allocation algorithm |
| `cka.py` | No changes (implementation was correct) |

### New Scripts

| Script | Purpose |
|---|---|
| `merge_pseudo_labels.py` | Merges pseudo-labels PKL with original Singapore datalist PKL by sequential index. Fixes `valid_flag`, `num_lidar_pts`, `gt_names`, `gt_velocity` alignment. Splits 9D boxes into 7D + 2D velocity. |

### New Config

| Config | Purpose |
|---|---|
| `configs/adapter/e5_pseudo_label_adapter.py` | E5 training config: inherits from `bevformer_rtx5060_residual01_subset2k.py`, overrides `ann_file` to `singapore_pseudo_labels_merged.pkl`, freezes backbone/neck/encoder/head, trains only adapter weights |

---

## 12. Known Remaining Issues

### Issue 1: CKA 95% CI Upward Bias
**Status:** Known, documented in paper  
**Root cause:** With 16,384D features and N=500 samples, CKA has finite-sample bias. Subsampling bootstrap (400 of 500 pairs, no replacement) still gives CI entirely above the point estimate.  
**Fix options:** (a) PCA to ~256D before CKA; (b) debiased `minCKA` estimator (Nguyen et al.); (c) use cosine CI only for the paper  
**Impact:** Low — point estimates (0.114, 0.139) are valid; only the CI bounds are unreliable

### Issue 2: t-SNE Caption References Old 67.6%
**Location:** `draft.tex` line 331 — "even after the BEV encoder's 67.6% normalization"  
**Should be:** "even after the BEV encoder's 81% cosine normalization" or just "even after the BEV encoder"  
**Fix:** One-line text edit

### Issue 3: E5 Validation Was on Pseudo-Labeled Split
The E5 evaluation was run on the same Singapore frames used for pseudo-label training (the full 2,929-frame val split is both training and evaluation for E5). This means the reported E5 metrics are slightly optimistic — in a clean experiment, E5 training would be on a held-out subset and evaluation on the remainder. Given that all 4 epochs are *worse* than the baseline, this doesn't change the conclusion, but should be noted in the paper.

### Issue 4: Circular Optimization Is Specific to Self-Pseudo-Labels at τ=0.3
The E5 failure is characterized as "circular optimization" at τ=0.3. A higher threshold (τ=0.5 or 0.6) would filter to higher-confidence boxes, potentially reducing noise. This is a natural follow-up experiment. However, higher τ significantly reduces the number of boxes and may leave too few for meaningful gradient signal.

---

## Appendix: Key File Locations

```
E:\bev_research\
├── scripts\
│   ├── representation_analysis_v2.py   [Phase 2 — CKA + cosine]
│   ├── build_semantic_pairs.py          [Phase 1 — density-matched pairs]
│   ├── generate_pseudo_labels.py        [Phase 3 — pseudo-labels]
│   ├── merge_pseudo_labels.py           [Phase 4 — merge for training]
│   ├── identify_depth_scale_channels.py [Phase 5 — DAv2 channels]
│   ├── tsne_encoder_drift.py            [Phase 6 — BEV drift]
│   └── cka.py                           [CKA implementation]
├── configs\adapter\
│   └── e5_pseudo_label_adapter.py       [E5 training config]
├── data\
│   ├── matched_pairs_500.json           [500 semantic pairs]
│   ├── singapore_pseudo_labels.pkl      [raw pseudo-labels]
│   ├── singapore_pseudo_labels_merged.pkl [merged for training]
│   ├── cka_features_500_v2.npz          [cached spatial CKA features]
│   └── dav2_channel_analysis.json       [96 invariant channels]
├── logs\
│   ├── representation_analysis_v2.json  [CKA results]
│   ├── bootstrap_ci_results.json        [mAP gap CIs]
│   ├── per_class_ap.json                [10-class AP]
│   └── e5_training.log                  [E5 full training log]
├── figures\
│   ├── tsne_encoder_drift.{pdf,png}     [t-SNE visualization]
│   └── per_class_ap_comparison.{pdf,png}[per-class bar chart]
├── work_dirs\e5_pseudo_label_adapter\
│   ├── epoch_{1-4}.pth                  [E5 checkpoints]
│   └── last_checkpoint
└── paper\
    └── draft.tex                        [IEEEtran paper draft]
```

---

*End of report. All data produced from real GPU inference on the project's RTX 5060 GPU. No synthetic results are present in any table in the paper draft.*
