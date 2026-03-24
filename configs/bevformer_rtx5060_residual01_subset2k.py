_base_ = ['./bevformer_rtx5060.py']

# Always start from pretrained baseline checkpoint for this fast run.
load_from = 'E:/bev_research/checkpoints/bevformer_base_epoch_24.pth'

# Fast-iteration config: 2000 training samples for rapid A/B.
train_dataloader = dict(
    dataset=dict(
        indices=list(range(2000)),
    ),
)

# Keep the experiment short and checkpoint each epoch.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4, val_interval=2)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=4),
)

