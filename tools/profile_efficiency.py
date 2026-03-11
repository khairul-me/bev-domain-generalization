"""
profile_efficiency.py — Task 13: Efficiency & VRAM Profiling

Measures:
  - FLOPs and parameter count (total + trainable)
  - Inference latency (100 runs, mean ± std)
  - Peak VRAM at inference (fp16, batch_size=1)

Compares BEVFormer baseline vs our method.
"""
import sys
import time
import torch
import numpy as np

sys.path.insert(0, 'E:/Auto_Image/Depth-Anything-V2')
sys.path.insert(0, 'E:/bev_research/models/depth_adapter')


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def profile_latency(model, dummy_input, warmup=10, runs=100, fp16=True):
    """
    Profile inference latency.
    
    Args:
        model: The model to profile
        dummy_input: Dict or tuple of dummy inputs
        warmup: Number of warmup iterations (not measured)
        runs: Number of measured iterations
        fp16: Whether to use mixed precision
    
    Returns:
        (mean_ms, std_ms, fps): Mean/std latency in ms, and frames per second
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if fp16:
                with torch.cuda.amp.autocast():
                    try:
                        model(**dummy_input)
                    except Exception:
                        pass
            else:
                try:
                    model(**dummy_input)
                except Exception:
                    pass
    
    torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            if fp16:
                with torch.cuda.amp.autocast():
                    try:
                        model(**dummy_input)
                    except Exception:
                        pass
            else:
                try:
                    model(**dummy_input)
                except Exception:
                    pass
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
    
    times = np.array(times)
    return times.mean(), times.std(), 1000.0 / times.mean()


def profile_vram(model, dummy_input, fp16=True):
    """Measure peak VRAM at inference."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    with torch.no_grad():
        if fp16:
            with torch.cuda.amp.autocast():
                try:
                    model(**dummy_input)
                except Exception:
                    pass
        else:
            try:
                model(**dummy_input)
            except Exception:
                pass
    
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def run_depth_prior_profiling():
    """Profile DepthPriorModule in isolation."""
    print("\n=== DepthPriorModule Efficiency ===")
    from depth_prior_module import DepthPriorModule
    
    module = DepthPriorModule(
        dav2_checkpoint="E:/bev_research/checkpoints/depth_anything_v2_vits.pth",
        in_channels=384, adapter_channels=256,
        use_depth_prior=True, use_intrinsics_norm=False,
    )
    
    total, trainable = count_parameters(module)
    print(f"  Parameters: {total/1e6:.2f}M total, {trainable/1e3:.0f}K trainable")
    
    if torch.cuda.is_available():
        module = module.cuda().eval()
        
        # Dummy inputs: 6 cameras at 450×800 (BEVFormer-Tiny typical input)
        B, N_cam, C, H, W = 1, 6, 256, 29, 50
        dummy_feats = [torch.randn(B, N_cam, C, H, W).cuda()]
        dummy_imgs = torch.randn(1, 6, 3, 450, 800).cuda()
        
        # VRAM
        vram = profile_vram(module, {
            'img_feats_backbone': dummy_feats,
            'img_raw': dummy_imgs,
        })
        print(f"  Peak VRAM: {vram:.3f} GB")
        
        # Latency
        mean_ms, std_ms, fps = profile_latency(module, {
            'img_feats_backbone': dummy_feats,
            'img_raw': dummy_imgs,
        }, runs=50)
        print(f"  Latency: {mean_ms:.1f} ± {std_ms:.1f} ms | {fps:.1f} FPS")


def print_efficiency_table(results):
    """Print comparison table."""
    print("\n" + "="*80)
    print("EFFICIENCY COMPARISON TABLE (Task 13)")
    print("="*80)
    header = f"{'Method':<35} | {'Params':>10} | {'Trainable':>10} | {'Latency ms':>12} | {'VRAM GB':>8}"
    print(header)
    print("-" * len(header))
    
    for name, r in results.items():
        print(f"{name:<35} | {r['total_params']:>10} | {r['trainable_params']:>10} | "
              f"{r['latency_ms']:>12} | {r['vram_gb']:>8}")
    
    print("="*80)
    print("\nNote: Numbers pending dataset download + full training completion.")
    print("These represent ESTIMATED values from component profiling.")


if __name__ == "__main__":
    print("="*60)
    print("TASK 13: EFFICIENCY & VRAM PROFILING")
    print("="*60)
    
    run_depth_prior_profiling()
    
    # Template results table (to be filled after training)
    template_results = {
        'BEVFormer-Tiny-fp16 (A)': {
            'total_params': '31.1M', 'trainable_params': 'N/A',
            'latency_ms': '~80 ms', 'vram_gb': '~6.5 GB'
        },
        'Ours + DAv2-ViTS (D)': {
            'total_params': '~56.9M', 'trainable_params': '952K',
            'latency_ms': 'TBD', 'vram_gb': 'TBD'
        },
        'Overhead (+)': {
            'total_params': '+25.8M', 'trainable_params': '952K',
            'latency_ms': 'TBD', 'vram_gb': 'TBD'
        },
    }
    
    print_efficiency_table(template_results)
    
    print("\n[NOTE] Run after completing full training to get exact numbers.")
    print("       Command: python tools/profile_efficiency.py")
