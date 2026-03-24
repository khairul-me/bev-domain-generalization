import argparse
import math

import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser(description="One-step gradient probe for adapter training.")
    parser.add_argument(
        "--config",
        default="E:/bev_research/configs/bevformer_rtx5060.py",
        help="Path to config file",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(
        {
            "train_dataloader.batch_size": 1,
            "train_dataloader.num_workers": 0,
            "train_dataloader.persistent_workers": False,
            "train_dataloader.dataset.queue_length": 4,
            "model.freeze_bevformer": True,
            "model.depth_adapter": {
                "enabled": True,
                "depth_repo_path": "E:/Auto_Image/bev_research/Depth-Anything-V2",
                "ckpt_path": "E:/bev_research/checkpoints/depth_anything_v2_vits.pth",
                "dav2_encoder": "vits",
                "intermediate_layers": [11],
                "adapter_hidden_ratio": 1.0,
                "residual_scale": 1.0,
                "input_size": 308,
                "inject_levels": [0],
                "use_amp": True,
                "feature_channels": [256, 256, 256, 256],
            },
        }
    )
    cfg.work_dir = "E:/bev_research/experiments/E3_adapter/gradient_probe"

    runner = Runner.from_cfg(cfg)
    model = runner.model
    model.train()
    optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
    optimizer = optim_wrapper.optimizer

    data_batch = next(iter(runner.train_dataloader))
    processed = model.data_preprocessor(data_batch, training=True)

    optim_wrapper.zero_grad()
    losses = model.loss(processed["inputs"], processed["data_samples"])
    loss = sum(v for v in losses.values() if torch.is_tensor(v))

    if isinstance(optim_wrapper.__class__.__name__, str) and "AmpOptimWrapper" in optim_wrapper.__class__.__name__:
        optim_wrapper.backward(loss)
    else:
        loss.backward()

    print("=== Raw Grad Norm Per Parameter ===")
    total_norm_sq = 0.0
    grad_count = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2).item()
        total_norm_sq += param_norm**2
        grad_count += 1
        print(f"{name}: grad_norm={param_norm:.10f}")

    total_norm = math.sqrt(total_norm_sq)
    print(f"Total grad norm: {total_norm:.12f}")
    print(f"Parameters with grads: {grad_count}")

    print("\n=== Optimizer Param Groups ===")
    for i, pg in enumerate(optimizer.param_groups):
        print(f"Group {i}: {len(pg['params'])} tensors, lr={pg.get('lr')}")

    trainable = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    optim_params = set()
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            optim_params.add(id(p))

    missing_trainable = [n for pid, n in trainable.items() if pid not in optim_params]
    print(f"\nTrainable params in model: {len(trainable)}")
    print(f"Trainable params missing from optimizer: {len(missing_trainable)}")
    for n in missing_trainable:
        print(f"MISSING: {n}")

    print("\n=== AMP Loss Scale ===")
    scale_value = None
    if hasattr(optim_wrapper, "loss_scaler"):
        scaler = optim_wrapper.loss_scaler
        get_scale = getattr(scaler, "get_scale", None)
        if callable(get_scale):
            scale_value = float(get_scale())
            print(f"Loss scale: {scale_value}")
        else:
            print(f"Loss scaler object: {scaler}")
    elif hasattr(optim_wrapper, "loss_scale"):
        scale_value = float(optim_wrapper.loss_scale)
        print(f"Loss scale: {scale_value}")
    else:
        print("Loss scale: not available (non-AMP wrapper)")

    if scale_value is not None and scale_value > 0:
        print(f"Estimated unscaled total grad norm: {total_norm / scale_value:.12f}")


if __name__ == "__main__":
    main()

