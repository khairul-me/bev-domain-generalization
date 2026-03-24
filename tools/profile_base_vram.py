import sys

import torch

sys.path.insert(0, r'E:\Auto_Image\bev_research\mmdetection3d')
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS
import projects.mmdet3d_plugin  # noqa: F401

config_path = r'E:\bev_research\configs\bevformer_rtx5060.py'
ckpt_path = r'E:\bev_research\checkpoints\bevformer_base_epoch_24.pth'

print('Loading config...')
cfg = Config.fromfile(config_path)
cfg.model.train_cfg = None
init_default_scope('mmdet3d')

print('Building model...')
model = MODELS.build(cfg.model)
model = model.cuda().eval()

print('Loading checkpoint...')
load_checkpoint(model, ckpt_path, map_location='cuda')

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

print('Running warmup forward pass...')
with torch.no_grad(), torch.cuda.amp.autocast():
    dummy_img = torch.randn(1, 1, 6, 3, 928, 1600).cuda().half()
    img_metas = [[{
        'scene_token': 'dummy',
        'can_bus': [0] * 18,
        'lidar2img': [torch.eye(4).numpy() for _ in range(6)],
        'img_shape': [(928, 1600, 3)] * 6,
        'ori_shape': [(900, 1600, 3)] * 6,
        'pad_shape': [(928, 1600, 3)] * 6,
        'scale_factor': 1.0,
        'flip': False,
        'box_type_3d': None,
    }]]
    try:
        model.forward_test(dummy_img, img_metas=img_metas)
    except Exception as e:
        print(f'Forward pass error (may be OK for dummy input): {e}')

peak_vram = torch.cuda.max_memory_allocated() / 1e9
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

print('\n--- VRAM Profile ---')
print(f'Peak VRAM used:  {peak_vram:.2f} GB')
print(f'Total VRAM:      {total_vram:.2f} GB')
print(f'Headroom left:   {total_vram - peak_vram:.2f} GB')

if peak_vram < 14.0:
    print('\nRESULT: Base checkpoint FITS in VRAM at fp16.')
    print(
        f'        You have {total_vram - peak_vram:.1f} GB headroom for the depth adapter.'
    )
    print('        Proceed with BEVFormer-Base as your baseline.')
else:
    print('\nRESULT: Base checkpoint EXCEEDS safe VRAM limit.')
    print('        Switch to BEVFormer-Small or reduce input resolution.')
    print('        Contact guide author before proceeding.')
