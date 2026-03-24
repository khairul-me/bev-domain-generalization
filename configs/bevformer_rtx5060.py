# BEVFormer Base config for RTX 5060 Ti
# Dataset ann_files use LEGACY format (infos key) for CustomNuScenesDataset
# Evaluator ann_files use METRIC format (data_list key) for NuScenesMetric

_base_ = [
    'E:/Auto_Image/bev_research/mmdetection3d/projects/configs/bevformer/bevformer_base.py',
]

default_scope = 'mmdet3d'

custom_imports = dict(
    imports=[
        'projects.mmdet3d_plugin',
        'projects.mmdet3d_plugin.datasets.nuscenes_dataset',
        'projects.mmdet3d_plugin.datasets.pipelines.transform_3d',
        'projects.mmdet3d_plugin.bevformer.detectors.bevformer',
        'projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head',
        'projects.mmdet3d_plugin.bevformer.modules.transformer',
        'projects.mmdet3d_plugin.bevformer.modules.encoder',
        'projects.mmdet3d_plugin.bevformer.modules.decoder',
        'projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention',
        'projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention',
        'projects.mmdet3d_plugin.bevformer.hooks.adapter_debug_hook',
        'projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d',
        'projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder',
        'projects.mmdet3d_plugin.core.bbox.match_costs.match_cost',
        'projects.mmdet3d_plugin.bevformer_nuscenes_metric',
    ],
    allow_failed_imports=False,
)

load_from = 'E:/bev_research/checkpoints/bevformer_base_epoch_24.pth'

# ── BEV resolution ───────────────────────────────────────────────────────────
bev_h_ = 200
bev_w_ = 200

model = dict(
    depth_adapter=dict(
        enabled=True,
        depth_repo_path='E:/Auto_Image/bev_research/Depth-Anything-V2',
        ckpt_path='E:/bev_research/checkpoints/depth_anything_v2_vits.pth',
        dav2_encoder='vits',
        intermediate_layers=[11],
        adapter_hidden_ratio=1.0,
        residual_scale=0.01,
        input_size=308,
        inject_levels=[0],
        use_amp=True,
        feature_channels=[256, 256, 256, 256],
        consistency_weight=0.1,
        consistency_aug=dict(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.1,
        ),
    ),
    freeze_bevformer=True,
    img_backbone=dict(
        type='mmdet.ResNet',
        with_cp=True,
    ),
    img_neck=dict(type='mmdet.FPN'),
    pts_bbox_head=dict(
        bev_h=bev_h_,
        bev_w=bev_w_,
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(type='MyCustomBaseTransformerLayer'),
            ),
        ),
    ),
)

# ── Dataset paths ─────────────────────────────────────────────────────────────
data_root = 'C:/datasets/nuscenes/'

# ── Optimizer with AMP ────────────────────────────────────────────────────────
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'depth_adapter': dict(lr_mult=0.001),
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
    loss_scale='dynamic',
)

# ── Training loop ─────────────────────────────────────────────────────────────
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Hooks ─────────────────────────────────────────────────────────────────────
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=4, max_keep_ckpts=6),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(type='AdapterDebugHook', interval=50, run_id='stabilized-precheck', log_path='E:/Auto_Image/debug-2a5848.log'),
]

# ── Shared pipeline / class definitions ───────────────────────────────────────
_classes_ = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

_modality_ = dict(
    use_lidar=False, use_camera=True, use_radar=False,
    use_map=False, use_external=True,
)

# ── Train dataloader ──────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CustomNuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(type='LoadAnnotations3D',
                 with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
            dict(type='ObjectRangeFilter',
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(type='ObjectNameFilter', classes=_classes_),
            dict(type='NormalizeMultiviewImage',
                 mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0],
                 to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='Pack3DDetInputs',
                 keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
                 meta_keys=[
                     'filename', 'ori_shape', 'img_shape', 'lidar2img',
                     'cam2img', 'lidar2cam', 'pad_shape', 'scale_factor',
                     'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                     'scene_token', 'can_bus', 'sample_idx', 'prev_bev_exists',
                 ]),
        ],
        metainfo=dict(classes=_classes_),
        modality=_modality_,
        test_mode=False,
        use_valid_flag=True,
        serialize_data=False,
        bev_size=(bev_h_, bev_w_),
        queue_length=4,
        box_type_3d='LiDAR',
    ),
)

# ── Val dataloader ────────────────────────────────────────────────────────────
# Dataset reads LEGACY PKL (has 'infos' key with all fields get_data_info needs)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CustomNuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='NormalizeMultiviewImage',
                 mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0],
                 to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='Pack3DDetInputs',
                 keys=['img'],
                 meta_keys=[
                     'filename', 'ori_shape', 'img_shape', 'lidar2img',
                     'cam2img', 'lidar2cam', 'pad_shape', 'scale_factor',
                     'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                     'scene_token', 'can_bus', 'sample_idx', 'prev_bev_exists',
                 ]),
        ],
        metainfo=dict(classes=_classes_),
        modality=_modality_,
        test_mode=True,
        serialize_data=False,
        bev_size=(bev_h_, bev_w_),
        queue_length=1,
        box_type_3d='LiDAR',
    ),
)

test_dataloader = val_dataloader

# ── Evaluator reads METRIC PKL (has 'data_list' key with new-format fields) ──
val_evaluator = dict(
    type='BEVFormerNuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_temporal_val_metric.pkl',
    metric='bbox',
)
test_evaluator = val_evaluator
