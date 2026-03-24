# Depth Estimation Chain in BEVFormer — Step 2.6

**Purpose:** Map data flow from raw camera image → depth feature → BEV feature → detection output.  
**Use:** Informs architecture design in Task 7.

---

## Data Flow Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        BEVFormer Forward Pass                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

  Multi-Camera Images (6×)                    BEV Grid Queries (H×W×C)
         │                                            │
         ▼                                            │
  ┌──────────────┐                                    │
  │ Image        │  ResNet-50 / Backbone               │
  │ Backbone     │  → 2D feature maps F_cam            │
  └──────┬───────┘                                    │
         │                                            │
         │  F_cam: [B, 6, C, H', W']                  │
         │                                            │
         ▼                                            ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  SPATIAL CROSS-ATTENTION (where depth is implicitly estimated)                │
  │                                                                               │
  │  1. BEV query Q_bev at grid (x, y) → 3D reference points at multiple depths   │
  │     (e.g., d ∈ {1m, 2m, ..., 50m})                                            │
  │                                                                               │
  │  2. Project 3D points to 2D via camera intrinsics K, extrinsics T:             │
  │     p_2d = K @ T @ p_3d                                                        │
  │     ★ ASSUMPTION: K (focal, principal point) is nuScenes-specific             │
  │                                                                               │
  │  3. Sample features from F_cam at p_2d (bilinear interpolation)               │
  │                                                                               │
  │  4. Attention weights over depth bins → implicit depth distribution            │
  │     (no explicit depth value; learned through attention)                       │
  │                                                                               │
  │  5. Weighted sum → BEV feature at (x, y)                                       │
  └──────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────┐
  │ Temporal     │  Fuse history BEV (recurrent)
  │ Self-Attn    │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Detection    │  3D bboxes, classes
  │ Head         │
  └──────────────┘
```

---

## Where Depth is Handled

| Location | Type | Description |
|----------|------|-------------|
| **3D reference points** | Explicit | Predefined depth bins (e.g., 1–50 m) — fixed, not learned |
| **Projection** | Explicit | Uses K, T — **assumes nuScenes camera model** |
| **Attention over depth** | Implicit | Network learns which depth bin to attend to — **this is the learned depth** |
| **No explicit depth head** | — | BEVFormer does not predict depth maps |

---

## Camera Intrinsic Assumptions

| Parameter | nuScenes | KITTI | Impact |
|-----------|----------|-------|--------|
| Focal length | ~1266 px | ~721 px | Same 3D point → different 2D location |
| Principal point | (800, 450) | (621, 187) | Shift in sampling grid |
| Resolution | 1600×900 | 1242×375 | Different feature map scale |
| Ego height | ~1.5 m | ~1.6 m | Slight change in BEV grid |

**Result:** 3D reference points project to wrong 2D locations on KITTI → wrong features sampled → wrong implicit depth → failed detections.

---

## What Changes Between nuScenes and KITTI

1. **Projection mismatch:** `p_2d = K @ T @ p_3d` uses nuScenes K; KITTI has different K → wrong `p_2d`.
2. **Feature scale:** Different resolution → different receptive fields.
3. **Depth distribution:** Object distances, road layout differ → learned attention over depth bins is miscalibrated.
4. **Temporal:** Different frame rates, motion patterns.

---

## Where Depth Anything V2 Plugs In

```
                    BEFORE (BEVFormer)                    AFTER (Ours)
                    
  Image → Backbone → F_cam ──────────────────┐     Image → Backbone → F_cam
                                              │              │
  BEV queries → project → sample F_cam ───────┼────►        │
  (implicit depth from attention)             │     Image ──┼──► DAv2 (frozen) → depth features
                                              │              │         │
                                              │              └─────────┼──► Adapter → F_depth
                                              │                        │
                                              └────────────────────────┼──► Concat/add F_cam + F_depth
                                                                        │
                                                                        ▼
                                                              BEV queries sample (F_cam + F_depth)
                                                              (DAv2 provides domain-agnostic depth prior)
```

**Key:** DAv2 gives depth features without relying on nuScenes-specific projection. Intrinsics normalization (Task 7.4) warps image to canonical view before DAv2.

---

## Diagram File

A visual diagram can be drawn and saved as `models/depth_adapter/architecture_diagram.png` in Task 7.
