# bevformer_depth_prior_finetune.py — Task 8 Training Config
# Fine-tunes ONLY the depth adapter + detection head
# Frozen: DAv2 encoder, ResNet-50 backbone
# Trainable: adapter (~952K params), BEV encoder attention, detection head

_base_ = [
    'E:/Auto_Image/BEVFormer/projects/configs/bevformer_fp16/bevformer_tiny_fp16.py',
]

# ── Model type override ──────────────────────────────────────────────────────
# Use our extended detector instead of BEVFormer_fp16
model = dict(
    type='BEVFormerWithDepthPrior',
    use_depth_prior=True,
    dav2_checkpoint='E:/bev_research/checkpoints/depth_anything_v2_vits.pth',
    depth_adapter_channels=256,
    use_intrinsics_norm=True,
    depth_fusion_mode='add',
    # Inherit all other args from bevformer_tiny_fp16
)

# ── Dataset paths ────────────────────────────────────────────────────────────
data_root = 'C:/datasets/nuscenes/'

data = dict(
    samples_per_gpu=1,
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

# ── Optimizer — layer-wise LR ─────────────────────────────────────────────
# Adapter gets full LR, frozen parts get 0.0
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            # Completely frozen components
            'depth_prior.dav2': dict(lr_mult=0.0, decay_mult=0.0),
            'img_backbone': dict(lr_mult=0.0, decay_mult=0.0),
            # Trainable components
            'depth_prior.adapter': dict(lr_mult=1.0),
            'depth_prior.depth_scale': dict(lr_mult=1.0),
            # BEV encoder attention — optional fine-tuning
            'pts_bbox_head.transformer.encoder': dict(lr_mult=0.1),
            # Detection head — full LR
            'pts_bbox_head': dict(lr_mult=1.0),
        }
    ),
)

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
    type='Fp16OptimizerHook',
    loss_scale=512.,
)

# ── Learning rate schedule ──────────────────────────────────────────────────
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

# ── Training schedule ───────────────────────────────────────────────────────
total_epochs = 24
fp16 = dict(loss_scale=512.)

# ── Checkpoint saving ───────────────────────────────────────────────────────
checkpoint_config = dict(interval=4, max_keep_ckpts=6)
# Saves every 4 epochs, keeps last 6. Best checkpoint selected by val NDS in Task 8.6.

# ── Evaluation ─────────────────────────────────────────────────────────────
evaluation = dict(interval=4)  # Eval every 4 epochs (not every epoch — saves time)

# ── Logging ────────────────────────────────────────────────────────────────
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

# ── Reproducibility ─────────────────────────────────────────────────────────
seed = 42
deterministic = True

# ── Training command ───────────────────────────────────────────────────────
# cd E:\Auto_Image\BEVFormer
# python tools/train.py \
#     E:/bev_research/configs/bevformer_depth_prior_finetune.py \
#     --work-dir E:/bev_research/experiments/with_adapter/full_train \
#     --seed 42
