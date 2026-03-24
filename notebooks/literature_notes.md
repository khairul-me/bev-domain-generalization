# Literature Notes — Task 2

Structured notes for each paper. Fill in details as you read.

---

## Paper Links (Corrected arXiv IDs)

| # | Paper | arXiv | PDF |
|---|-------|-------|-----|
| 1 | BEVFormer | [2203.17270](https://arxiv.org/abs/2203.17270) | [PDF](https://arxiv.org/pdf/2203.17270) |
| 2 | BEVDepth | [2206.10092](https://arxiv.org/abs/2206.10092) | [PDF](https://arxiv.org/pdf/2206.10092) |
| 3 | StreamPETR | [2303.11926](https://arxiv.org/abs/2303.11926) | [PDF](https://arxiv.org/pdf/2303.11926) |
| 4 | DETR3D | [2110.06922](https://arxiv.org/abs/2110.06922) | [PDF](https://arxiv.org/pdf/2110.06922) |
| 5 | Depth Anything V2 | [2406.09414](https://arxiv.org/abs/2406.09414) | [PDF](https://arxiv.org/pdf/2406.09414) |
| 6 | DA-BEV (domain adapt) | Search "DA-BEV domain adaptation BEV" / [OpenReview](https://openreview.net/forum?id=RfxiwnCcg8) | |
| 7 | OpenAD Benchmark | [2411.17761](https://arxiv.org/abs/2411.17761) | [PDF](https://arxiv.org/pdf/2411.17761) |
| 8 | nuScenes dataset | [1903.11027](https://arxiv.org/abs/1903.11027) | [PDF](https://arxiv.org/pdf/1903.11027) |
| 9 | KITTI benchmark | [Geiger et al. CVPR 2012](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf) — Section 3 | |
| 10 | Tent (TTA) | [2006.10726](https://arxiv.org/abs/2006.10726) | [PDF](https://arxiv.org/pdf/2006.10726) |

---

## 1. BEVFormer (Li et al., ECCV 2022)

**Abstract summary:** Unified BEV representations via spatiotemporal transformers. Grid-shaped BEV queries + spatial cross-attention (extract from multi-camera) + temporal self-attention (fuse history). 56.9% NDS on nuScenes test.

| Field | Notes |
|-------|-------|
| **Problem** | 3D perception from multi-camera for autonomous driving |
| **Method (depth)** | *Implicit* depth via spatial cross-attention — BEV queries sample image features at projected 3D reference points; depth is learned implicitly through attention weights, not explicit depth prediction |
| **Results** | 56.9% NDS nuScenes test; 9 pts over prior; on par with LiDAR |
| **Limitation** | No explicit depth; depth implicitly tied to training domain (nuScenes intrinsics, camera setup) |
| **Relevance** | Baseline; depth is the bottleneck — implicit depth fails cross-domain |

---

## 2. BEVDepth (Li et al., AAAI 2023)

**Abstract summary:** Explicit depth supervision for camera BEV detection. Camera-awareness depth module + Depth Refinement Module. 60.9% NDS nuScenes.

| Field | Notes |
|-------|-------|
| **Problem** | Depth estimation in BEV methods is inadequate; depth is essential for 3D detection |
| **Method (depth)** | *Explicit* LiDAR-supervised depth; camera-awareness module; depth refinement for imprecise unprojection |
| **Results** | 60.9% NDS nuScenes; first camera model to reach 60% NDS |
| **Limitation** | Requires LiDAR for depth supervision — domain-specific; depth prior tied to nuScenes |
| **Relevance** | **Core depth bottleneck** — LiDAR-supervised depth does not generalize to KITTI/other domains |

---

## 3. StreamPETR (Wang et al., ICCV 2023)

**Abstract summary:** Object-centric temporal modeling for multi-view 3D detection. Sparse queries + frame-by-frame propagation + motion-aware layer norm. 67.6% NDS, 65.3% AMOTA; first online camera method comparable to LiDAR.

| Field | Notes |
|-------|-------|
| **Problem** | Efficient long-sequence temporal modeling for multi-view 3D detection |
| **Method (depth)** | Built on PETR — implicit 3D position encoding; no explicit depth; object queries propagate temporally |
| **Results** | 67.6% NDS, 65.3% AMOTA nuScenes; lightweight 45.0% mAP @ 31.7 FPS |
| **Limitation** | Still implicit geometry; trained on nuScenes — cross-domain generalization not addressed |
| **Relevance** | Alternative baseline; same implicit depth / domain-specific geometry issue |

---

## 4. DETR3D (Wang et al., CoRL 2021)

**Abstract summary:** 3D detection via 3D-to-2D queries. Sparse 3D queries index 2D features via camera matrices. Top-down, no per-pixel depth, no NMS.

| Field | Notes |
|-------|-------|
| **Problem** | Multi-camera 3D detection without compounding depth errors |
| **Method (depth)** | No explicit depth; 3D queries project to 2D and sample features; depth implicit in query–feature association |
| **Results** | SOTA nuScenes at time |
| **Limitation** | Implicit depth; domain-specific |
| **Relevance** | Foundation for BEVFormer/PETR; establishes sparse query paradigm |

---

## 5. Depth Anything V2 (Yang et al., NeurIPS 2024)

**Abstract summary:** Strong monocular depth via (1) synthetic-only teacher, (2) scaled teacher, (3) pseudo-labeled real images. 25M–1.3B params; 10× faster than SD-based; strong generalization.

| Field | Notes |
|-------|-------|
| **Problem** | Building a powerful, generalizable monocular depth model |
| **Method (depth)** | Foundation model; trained on 595K synthetic + 62M real; diverse domains |
| **Results** | Fine, robust depth; efficient; metric depth via fine-tuning |
| **Limitation** | Monocular; metric scale may need calibration per domain |
| **Relevance** | **Our depth prior** — frozen DAv2 provides domain-agnostic depth features |

---

## 6. DA-BEV (ECCV 2024 — Domain Adaptation)

**Search:** "DA-BEV domain adaptation BEV" or OpenReview ECCV 2024.

| Field | Notes |
|-------|-------|
| **Problem** | Unsupervised domain adaptation for BEV perception |
| **Method** | Query-based adversarial learning + query-based self-training; image-view and BEV regularization |
| **Results** | Superior on multiple datasets/tasks |
| **Limitation** | Requires target-domain (unlabeled) data; not zero-shot; adaptation overhead |
| **Relevance** | Primary comparison; we aim zero-shot + TTA without target data |

---

## 7. OpenAD Benchmark (2024/2025)

**Abstract summary:** First open-world AD benchmark for 3D detection. 2000 scenes, 5 datasets (nuScenes, Argoverse2, KITTI, ONCE, Waymo), corner cases, MLLM annotation.

| Field | Notes |
|-------|-------|
| **Problem** | Lack of comprehensive open-world 3D perception benchmarks |
| **Method** | Corner-case pipeline + MLLM; unified format across datasets |
| **Results** | Evaluates scene generalization, cross-vehicle, open-vocab, corner cases |
| **Limitation** | New benchmark; baselines still evolving |
| **Relevance** | How cross-domain failure is measured; evaluation protocol |

---

## 8. nuScenes Dataset (Caesar et al., CVPR 2020)

**Abstract summary:** Multimodal AD dataset — 6 cameras, 5 radars, 1 LiDAR, 360°; 1000 scenes, 20s each; 23 classes, 8 attributes.

| Field | Notes |
|-------|-------|
| **Problem** | Need for large-scale multimodal AD data |
| **Method** | NDS = 1/10 (mAP + mATE + mASE + mAOE + mAVE + mAAE); weighted combination |
| **Results** | 7× annotations, 100× images vs KITTI |
| **Limitation** | Boston + Singapore; specific sensor setup |
| **Relevance** | In-domain benchmark; NDS formula for evaluation |

---

## 9. KITTI Benchmark (Geiger et al., CVPR 2012)

**Focus:** Section 3 — 3D AP evaluation (IoU thresholds, difficulty levels).

| Field | Notes |
|-------|-------|
| **Problem** | 3D object detection benchmark |
| **Method** | 3D AP at IoU 0.7; Easy/Moderate/Hard; Car, Pedestrian, Cyclist |
| **Results** | Standard for cross-domain evaluation |
| **Limitation** | Germany; different cameras, resolution, intrinsics vs nuScenes |
| **Relevance** | Cross-domain target; different focal length, resolution, ego-height |

---

## 10. Tent: Fully Test-Time Adaptation (Wang et al., ICLR 2021)

**Abstract summary:** Adapt at test time by entropy minimization. Update BatchNorm affine (γ, β) using prediction entropy. No labels; one epoch.

| Field | Notes |
|-------|-------|
| **Problem** | Generalize to new data at test time without labels |
| **Method** | Entropy minimization; BN affine params; batch statistics at test |
| **Results** | SOTA ImageNet-C; source-free DA (SVHN→MNIST, GTA→Cityscapes, VisDA-C) |
| **Limitation** | Requires BN; may need careful tuning |
| **Relevance** | **Our TTA method** — entropy minimization on adapter BN for cross-domain |

---

## Reading Progress

- [ ] 1. BEVFormer
- [ ] 2. BEVDepth
- [ ] 3. StreamPETR
- [ ] 4. DETR3D
- [ ] 5. Depth Anything V2
- [ ] 6. DA-BEV
- [ ] 7. OpenAD
- [ ] 8. nuScenes
- [ ] 9. KITTI (Sec 3)
- [ ] 10. Tent

---

## Key Takeaways for Your Work

1. **Depth bottleneck:** BEVFormer/BEVDepth/StreamPETR rely on implicit or LiDAR-supervised depth → domain-specific.
2. **Cross-domain failure:** Different intrinsics (nuScenes ~1266px vs KITTI ~721px), resolution, ego-height → depth distribution shift.
3. **DAv2 as prior:** Frozen Depth Anything V2 gives domain-agnostic depth features (595K+62M images).
4. **TTA:** Tent-style entropy minimization on adapter BN for residual domain gap.
5. **DA-BEV gap:** Needs target data; we aim zero-shot + TTA.
