# BEVFormer Base — RTX 5060 Ti smoke-test config
# Run from: cd E:\Auto_Image\bev_research\mmdetection3d
# Command:  python tools/train.py E:/bev_research/configs/bevformer_rtx5060.py --work-dir E:/bev_research/experiments/baseline/smoke_test

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
        'projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d',
        'projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder',
        'projects.mmdet3d_plugin.core.bbox.match_costs.match_cost',
        'projects.mmdet3d_plugin.bevformer_nuscenes_metric',
    ],
    allow_failed_imports=False,
)

load_from = 'E:/bev_research/checkpoints/bevformer_base_epoch_24.pth'

# ── Full-resolution BEVFormer-Base settings ─────────────────────────────────
bev_h_ = 200
bev_w_ = 200

model = dict(
    img_backbone=dict(
        type='mmdet.ResNet',
        with_cp=True,      # gradient checkpointing — saves ~30% VRAM
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

# ── Optimizer with AMP (actual fp16) ─────────────────────────────────────────
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
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
    loss_scale='dynamic',
)

# ── Training loop ─────────────────────────────────────────────────────────────
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=24)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Hooks ─────────────────────────────────────────────────────────────────────
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=4, max_keep_ckpts=6),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# ── Dataloader ────────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CustomNuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='Pack3DDetInputs',
                keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'cam2img', 'lidar2cam', 'pad_shape', 'scale_factor',
                    'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                    'scene_token', 'can_bus', 'sample_idx',
                    'prev_bev_exists',
                ])
        ],
        metainfo=dict(
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ]),
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        use_valid_flag=True,
        serialize_data=False,
        bev_size=(bev_h_, bev_w_),
        queue_length=4,
        box_type_3d='LiDAR',
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CustomNuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val_singapore_datalist.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='Pack3DDetInputs',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'cam2img', 'lidar2cam', 'pad_shape', 'scale_factor',
                    'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                    'scene_token', 'can_bus', 'sample_idx',
                    'prev_bev_exists',
                ])
        ],
        metainfo=dict(
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ]),
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        serialize_data=False,
        bev_size=(bev_h_, bev_w_),
        queue_length=1,
        box_type_3d='LiDAR',
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='BEVFormerNuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_temporal_val_singapore_datalist.pkl',
    metric='bbox',
)
test_evaluator = val_evaluator
