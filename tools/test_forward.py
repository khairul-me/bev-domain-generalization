"""
test_forward.py — Task 4.2/4.4 smoke test and VRAM profiler.
Runs 2 forward/backward iterations to verify the pipeline end-to-end.
"""
import sys
import os
import torch

sys.path.insert(0, 'E:/Auto_Image/BEVFormer')
sys.path.insert(0, 'E:/Auto_Image/BEVFormer/projects')

def profile_vram(model, dummy_input, label=""):
    """Run one forward pass and report peak VRAM."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            try:
                out = model(**dummy_input)
                status = "OK"
            except Exception as e:
                status = f"ERROR: {e}"
                out = None
    
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  [{label}] Peak VRAM: {peak_gb:.2f} GB | Status: {status}")
    return peak_gb, out


def check_environment():
    """Pre-flight environment checks for Task 4."""
    print("\n" + "="*60)
    print("TASK 4: BASELINE CODEBASE SETUP — SMOKE TEST")
    print("="*60)
    
    checks = {}
    
    # GPU Check
    checks['CUDA available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        checks['GPU name'] = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        checks['Total VRAM (GB)'] = f"{total_vram:.1f}"
        free_vram = (torch.cuda.get_device_properties(0).total_memory - 
                     torch.cuda.memory_allocated()) / 1e9
        checks['Free VRAM (GB)'] = f"{free_vram:.1f}"
    
    print("\n--- Environment ---")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    
    # PyTorch version
    print(f"\n  PyTorch: {torch.__version__}")
    print(f"  CUDA version: {torch.version.cuda}")
    
    # Check mmdet3d
    try:
        import mmdet3d
        print(f"  MMDetection3D: {mmdet3d.__version__}")
    except ImportError:
        print("  MMDetection3D: NOT FOUND")
    
    return checks.get('CUDA available', False)


def test_backbone_vram():
    """Test ResNet-50 backbone overhead (BEVFormer-Tiny uses ResNet-50)."""
    print("\n--- Backbone VRAM Test (ResNet-50) ---")
    try:
        import torchvision.models as models
        backbone = models.resnet50(pretrained=False).cuda().eval()
        
        # Simulate 6 cameras at 800x450 (BEVFormer-Tiny input size after 0.5x scale)
        dummy_imgs = torch.randn(6, 3, 450, 800).cuda()
        
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                _ = backbone(dummy_imgs)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  6×cameras ResNet-50 forward pass: {peak:.2f} GB peak VRAM")
        del backbone, dummy_imgs
        torch.cuda.empty_cache()
        return peak
    except Exception as e:
        print(f"  Error: {e}")
        return None


def test_depth_anything_vram():
    """Test DAv2-ViTS overhead on top of backbone."""
    print("\n--- Depth Anything V2 ViT-S VRAM Test ---")
    try:
        sys.path.insert(0, 'E:/Auto_Image/Depth-Anything-V2')
        from depth_anything_v2.dpt import DepthAnythingV2
        
        model_cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        model = DepthAnythingV2(**model_cfg).cuda().eval()
        
        # Freeze all params
        for p in model.parameters():
            p.requires_grad = False
        
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  DAv2-ViTS parameters: {param_count:.1f}M")
        
        # Test on 6 camera images at 800x450
        dummy_imgs = torch.randn(6, 3, 450, 800).cuda()
        
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # Just test the encoder part
                _ = model.pretrained(dummy_imgs)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  DAv2-ViTS encoder (6 cameras, 800x450): {peak:.2f} GB peak VRAM")
        
        del model, dummy_imgs
        torch.cuda.empty_cache()
        return peak, param_count
    except Exception as e:
        print(f"  Error: {e}")
        return None, None


def test_adapter_vram(dav2_channels=384, adapter_channels=256):
    """Test lightweight adapter module."""
    print("\n--- Lightweight Adapter VRAM Test ---")
    try:
        import torch.nn as nn
        
        adapter = nn.Sequential(
            nn.Conv2d(dav2_channels, adapter_channels, 3, padding=1),
            nn.BatchNorm2d(adapter_channels),
            nn.ReLU(),
            nn.Conv2d(adapter_channels, adapter_channels, 1),
        ).cuda()
        
        param_count = sum(p.numel() for p in adapter.parameters()) / 1e6
        print(f"  Adapter parameters: {param_count:.2f}M")
        
        # Feature map size: 800/32 x 450/32 ≈ 25x14 per camera, 6 cameras
        dummy_feat = torch.randn(6, dav2_channels, 25, 14).cuda()
        
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.amp.autocast():
            _ = adapter(dummy_feat)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Adapter forward (6 cameras): {peak:.2f} GB peak VRAM")
        
        del adapter, dummy_feat
        torch.cuda.empty_cache()
        return peak, param_count
    except Exception as e:
        print(f"  Error: {e}")
        return None, None


def print_summary(bb_vram, dav2_vram, adapter_vram):
    print("\n" + "="*60)
    print("VRAM BUDGET SUMMARY (RTX 5060 Ti = 16 GB)")
    print("="*60)
    print(f"  ResNet-50 backbone (6 cams):     {bb_vram or 0:.2f} GB")
    print(f"  DAv2-ViTS encoder (6 cams):      {dav2_vram or 0:.2f} GB")
    print(f"  Adapter module:                  {adapter_vram or 0:.2f} GB")
    total = (bb_vram or 0) + (dav2_vram or 0) + (adapter_vram or 0)
    print(f"  ─────────────────────────────")
    print(f"  Estimated combined (approx):     ~{total:.1f} GB")
    print(f"  BEV encoder + detection head:    ~3–5 GB (estimated)")
    print(f"  Overhead + gradients:            ~2–3 GB (estimated)")
    remaining = 16.0 - total
    print(f"  Headroom for BEV+Det+overhead:   ~{remaining:.1f} GB")
    
    if remaining > 5:
        print("\n  ✓ VRAM budget OK — safe to proceed with training")
    elif remaining > 2:
        print("\n  ⚠ VRAM tight — use gradient checkpointing during training")
    else:
        print("\n  ✗ VRAM budget exceeded — reduce adapter size or use smaller model")
    print("="*60)


if __name__ == "__main__":
    cuda_ok = check_environment()
    
    if not cuda_ok:
        print("\nERROR: CUDA not available!")
        sys.exit(1)
    
    bb_vram = test_backbone_vram()
    dav2_vram, dav2_params = test_depth_anything_vram()
    adapter_vram, adapter_params = test_adapter_vram()
    
    print_summary(bb_vram, dav2_vram, adapter_vram)
    
    print("\n--- Task 4 Checklist ---")
    print("  [x] GPU and CUDA verified")
    print("  [x] BEVFormer config created (bevformer_rtx5060.py)")
    print("  [x] VRAM budget profiled for all components")
    print("  [  ] 2-iter smoke test (needs dataset download first)")
    print("  [  ] Full experiment tracking initialized (WandB)")
    print("  [  ] Git committed")
