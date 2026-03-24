# Task 2 Reading Guide

**Estimated time:** 5–7 days  
**Order:** Read in the sequence below for maximum efficiency.

---

## Phase 1: Core BEV Pipeline (Days 1–2)

### 1. BEVFormer [2203.17270]
**Focus:** Spatial cross-attention, temporal self-attention, grid BEV queries  
**Key question:** Where and how is depth implicitly used?  
**Sections:** Method (3.2 Spatial Cross-Attention), Fig. 2

### 2. BEVDepth [2206.10092]
**Focus:** LiDAR-supervised depth as bottleneck  
**Key question:** Why does explicit depth help in-domain but fail cross-domain?  
**Sections:** Introduction (depth inadequacy), Camera-Aware Depth Module

### 3. StreamPETR [2303.11926] *(Note: pipeline lists 2303.11425 — use 2303.11926)*
**Focus:** Object-centric temporal propagation, memory queue  
**Key question:** How does it handle geometry without explicit depth?  
**Sections:** Object-Centric Temporal Mechanism

### 4. DETR3D [2110.06922]
**Focus:** 3D-to-2D query design  
**Key question:** How do sparse queries avoid depth prediction?  
**Sections:** Method, 3D-to-2D Feature Sampling

### 5. Depth Anything V2 [2406.09414]
**Focus:** Foundation depth model, generalization  
**Key question:** What makes it domain-agnostic?  
**Sections:** Training pipeline, model scales

---

## Phase 2: Cross-Domain & Evaluation (Days 3–4)

### 6. DA-BEV
**Search:** OpenReview ECCV 2024 "DA-BEV" or "domain adaptation BEV"  
**Focus:** Limitations of current domain adaptation for BEV  
**Key question:** Why do we need zero-shot + TTA instead?

### 7. OpenAD [2411.17761]
**Focus:** How cross-domain failure is measured  
**Key question:** Evaluation protocol for generalization  
**Sections:** Benchmark design, evaluation methodology

### 8. nuScenes [1903.11027]
**Focus:** NDS metric formula  
**Key question:** How is NDS computed?  
**Sections:** Metrics (NDS = 1/10 × (mAP + mATE + mASE + mAOE + mAVE + mAAE))

### 9. KITTI (Geiger et al., CVPR 2012)
**Focus:** Section 3 only — 3D AP evaluation  
**Key question:** IoU thresholds, difficulty levels  
**Link:** [cvlibs.net](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf)

---

## Phase 3: TTA (Day 5)

### 10. Tent [2006.10726]
**Focus:** Entropy minimization, BatchNorm adaptation  
**Key question:** How to adapt at test time without labels?  
**Sections:** Method, Algorithm 1

---

## After Reading

1. Complete structured notes in `literature_notes.md`
2. Review `depth_estimation_chain_bevformer.md` — add any corrections
3. Run through `TASK2_VERIFICATION_CHECKLIST.md`
4. Confirm alignment with overall plan (Tasks 1–2)
