"""Quick single-sample inference to diagnose mAP=0 issue."""
import sys, os, torch
sys.path.insert(0, r'E:\Auto_Image\bev_research\mmdetection3d')
os.chdir(r'E:\Auto_Image\bev_research\mmdetection3d')

from mmengine.config import Config
from mmdet3d.registry import DATASETS, MODELS
from mmengine.runner import load_checkpoint
import importlib

cfg = Config.fromfile(r'E:\bev_research\configs\bevformer_rtx5060.py')

# Trigger custom imports
for mod_path in cfg.custom_imports.imports:
    importlib.import_module(mod_path)

# Build dataset
dataset_cfg = cfg.val_dataloader.dataset
dataset = DATASETS.build(dataset_cfg)
print(f'Dataset size: {len(dataset)}')

# Get one sample
sample = dataset[0]
print(f'\nSample type: {type(sample).__name__}')

# Inspect sample structure
if hasattr(sample, 'keys'):
    for k, v in sample.items():
        if hasattr(v, 'shape'):
            print(f'  {k}: Tensor shape={v.shape}')
        elif hasattr(v, 'keys'):
            print(f'  {k}: dict keys={list(v.keys())[:5]}')
        else:
            print(f'  {k}: {type(v).__name__}')

# Check data_samples
if 'data_samples' in sample:
    ds = sample['data_samples']
    print(f'\ndata_samples type: {type(ds).__name__}')
    meta = ds.metainfo
    print(f'metainfo keys: {sorted(meta.keys())}')
    if 'lidar2img' in meta:
        l2i = meta['lidar2img']
        if isinstance(l2i, list):
            print(f'lidar2img: list of {len(l2i)} matrices')
            import numpy as np
            mat = np.array(l2i[0])
            print(f'  lidar2img[0] shape: {mat.shape}')
            print(f'  lidar2img[0]:\n{mat}')
    if 'can_bus' in meta:
        print(f'can_bus: {meta["can_bus"][:5]}...')
    if 'sample_idx' in meta:
        print(f'sample_idx: {meta["sample_idx"]}')

# Check images
if 'inputs' in sample:
    inputs = sample['inputs']
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f'\ninputs[{k}]: shape={v.shape}, dtype={v.dtype}')
            print(f'  min={v.min():.1f}, max={v.max():.1f}, mean={v.mean():.1f}')
        elif isinstance(v, list):
            print(f'\ninputs[{k}]: list of {len(v)}')
            if v and isinstance(v[0], torch.Tensor):
                print(f'  [0] shape={v[0].shape}')
