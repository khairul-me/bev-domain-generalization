_base_ = ['./bevformer_rtx5060.py']

# Start from pretrained baseline for controlled comparison.
load_from = 'E:/bev_research/checkpoints/bevformer_base_epoch_24.pth'

# Partial unfreeze setup:
# - depth_adapter: full base lr (2e-4), residual_scale=0.1 so delta is non-trivial
# - BEVFormer encoder: lr=1e-5 (0.05 × 2e-4) — small enough to be stable
# - backbone, neck, detection head: lr=0 (frozen via optimizer, requires_grad stays True)
# - consistency loss disabled — detection loss alone drives the training signal
model = dict(
    freeze_bevformer=False,
    partial_unfreeze_encoder=True,
    depth_adapter=dict(
        consistency_weight=0.0,
        residual_scale=0.1,
    ),
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.0),
            'pts_bbox_head.transformer.encoder': dict(lr_mult=0.05),
            'pts_bbox_head': dict(lr_mult=0.0),
            'depth_adapter': dict(lr_mult=1.0),
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Fast iteration config for quick decision.
train_dataloader = dict(
    dataset=dict(
        indices=list(range(2000)),
    )
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4, val_interval=2)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=4),
)
custom_hooks = [
    dict(
        type='AdapterDebugHook',
        interval=50,
        run_id='partial-unfreeze',
        log_path='E:/Auto_Image/debug-partial-unfreeze.log',
    ),
]

