"""
E6: Depth-scale-only channel ablation.

What this config does:
  - Identical to E3-B (alpha=0.1, Boston training subset, frozen BEVFormer)
  - EXCEPT: the adapter's first Conv2d takes only 96 depth-scale-invariant
    channels (out of 384 total DAv2 ViT-S channels), selected by correlation
    with log-depth-std across 100 calibration frames.

Scientific question answered:
  Does restricting injection to the 96 channels with the smallest cross-city
  depth-scale variation change the outcome vs. injecting all 384 channels?

Expected outcome:
  If E3-B failed because the wrong channels introduced city-specific noise,
  E6 should improve.
  If E3-B failed because Boston detection loss drives any adapter to zero
  output regardless of input channels, E6 should fail identically.

The second outcome (expected based on mechanistic analysis) directly confirms
Finding 4: the failure is in the supervision signal, not the channel selection.

Parameter count:
  E3-B: Conv2d(384, 256) + Conv2d(256, 256) = 98,560 + 65,792 = 164,352
  E6:   Conv2d(96,  256) + Conv2d(256, 256) =  24,832 + 65,792 =  90,624

Training command:
    conda activate bev310
    cd E:\Auto_Image\bev_research\mmdetection3d
    python tools\train.py E:\bev_research\configs\adapter\e6_depth_scale_channels.py `
        --work-dir E:\bev_research\work_dirs\e6_depth_scale_channels
"""

# Inherit from E3-B (residual01, Boston subset, frozen BEVFormer)
_base_ = ['../bevformer_rtx5060_residual01_subset2k.py']

# The 96 depth-scale-invariant DAv2 channel indices, ordered by |r| with log_std
# (descending). Selected by identify_depth_scale_channels.py, seed=42, n=100 frames.
_DEPTH_SCALE_CHANNELS = [
    374, 77, 294, 315, 36, 25, 362, 299, 328, 334, 103, 61, 278, 43, 215, 344,
    10, 50, 226, 160, 224, 130, 369, 355, 372, 49, 22, 286, 282, 272, 20, 321,
    142, 51, 287, 35, 276, 245, 242, 143, 268, 161, 234, 19, 37, 262, 327, 288,
    349, 316, 261, 357, 237, 364, 2, 211, 28, 139, 300, 380, 263, 24, 184, 96,
    102, 7, 217, 254, 174, 115, 232, 246, 128, 117, 81, 253, 257, 53, 42, 68,
    275, 30, 89, 255, 202, 200, 310, 319, 151, 360, 163, 301, 45, 356, 79, 312,
]

model = dict(
    freeze_bevformer=True,
    depth_adapter=dict(
        residual_scale=0.1,          # identical to E3-B
        consistency_weight=0.0,      # no consistency loss
        channel_indices=_DEPTH_SCALE_CHANNELS,   # <-- only change vs E3-B
    ),
)
