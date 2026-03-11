# BEVFormer-Tiny (fp16) — Optimized for RTX 5060 Ti (16GB VRAM)
# Inherits from bevformer_tiny_fp16.py, adjusts data paths + hardware settings
# Usage: python tools/train.py configs/bevformer_rtx5060.py

_base_ = [
    'E:/Auto_Image/BEVFormer/projects/configs/bevformer_fp16/bevformer_tiny_fp16.py',
]

# ── Dataset paths (pointing to C:\datasets\nuscenes) ────────────────────────
data_root = 'C:/datasets/nuscenes/'

data = dict(
    samples_per_gpu=1,      # Safe for 16GB; increase to 2 if VRAM allows
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
    ),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        samples_per_gpu=1,
    ),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
    ),
)

# ── Optimizer — conservative settings for fine-tuning ──────────────────────
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01,
)

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
    type='Fp16OptimizerHook',
    loss_scale=512.,
)

# ── fp16 (mixed precision for VRAM efficiency) ──────────────────────────────
fp16 = dict(loss_scale=512.)

# ── Logging and checkpointing ──────────────────────────────────────────────
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

checkpoint_config = dict(interval=4, max_keep_ckpts=6)  # Save every 4 epochs, keep 6

# ── Work dir hint ──────────────────────────────────────────────────────────
# python tools/train.py configs/bevformer_rtx5060.py --work-dir E:/bev_research/experiments/baseline
