_base_ = ['./bevformer_rtx5060.py']

# Run B: same setup as Run A, only residual scale changed.
model = dict(
    depth_adapter=dict(
        residual_scale=0.1,
    ),
)

# Keep more frequent checkpointing for future branch safety.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        max_keep_ckpts=6,
    ),
)

