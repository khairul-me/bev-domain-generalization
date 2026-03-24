import argparse
import json
import time

import torch
from mmengine.config import Config
from mmengine.runner import Runner

LOG_PATH = r"E:\Auto_Image\debug-2a5848.log"
SESSION_ID = "2a5848"


def dlog(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main():
    parser = argparse.ArgumentParser(description="One-batch adapter smoke check.")
    parser.add_argument("--enable-adapter", action="store_true")
    parser.add_argument("--freeze-bevformer", action="store_true")
    parser.add_argument("--run-id", type=str, default="manual")
    parser.add_argument("--queue-length", type=int, default=4)
    args = parser.parse_args()
    run_id = args.run_id

    cfg = Config.fromfile("E:/bev_research/configs/bevformer_rtx5060.py")
    overrides = {
        "train_dataloader.batch_size": 1,
        "train_dataloader.num_workers": 0,
        "train_dataloader.persistent_workers": False,
        "train_dataloader.dataset.queue_length": int(args.queue_length),
        "env_cfg.cudnn_benchmark": False,
    }
    if args.enable_adapter:
        overrides["model.depth_adapter"] = {
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
        }
    if args.freeze_bevformer:
        overrides["model.freeze_bevformer"] = True
    cfg.merge_from_dict(overrides)
    cfg.work_dir = "E:/bev_research/experiments/adapter_smoke_check"
    # region agent log
    dlog(
        run_id,
        "H1",
        "adapter_smoke_check.py:54",
        "cfg_overrides_applied",
        {
            "enable_adapter": bool(args.enable_adapter),
            "freeze_bevformer": bool(args.freeze_bevformer),
            "queue_length": int(cfg.train_dataloader.dataset.queue_length),
            "batch_size": int(cfg.train_dataloader.batch_size),
            "num_workers": int(cfg.train_dataloader.num_workers),
        },
    )
    # endregion

    runner = Runner.from_cfg(cfg)
    model = runner.model
    model.train()
    # region agent log
    dlog(
        run_id,
        "H3",
        "adapter_smoke_check.py:73",
        "model_built",
        {
            "has_img_backbone": hasattr(model, "img_backbone"),
            "has_pts_bbox_head": hasattr(model, "pts_bbox_head"),
            "device": str(next(model.parameters()).device),
        },
    )
    # endregion

    # Before first forward (adapter projection layers are lazy and not built yet)
    pre_trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable params before first forward:", len(pre_trainable))
    if pre_trainable:
        for n in pre_trainable[:20]:
            print("  ", n)

    # One-shot hooks to prove key modules execute and with what shapes.
    hook_flags = {"backbone": False, "transformer": False}

    def _shape(x):
        if torch.is_tensor(x):
            return list(x.shape)
        if isinstance(x, (list, tuple)):
            return [_shape(y) for y in x]
        if isinstance(x, dict):
            return {k: _shape(v) for k, v in x.items()}
        return str(type(x))

    def _backbone_hook(_module, _inp, out):
        if hook_flags["backbone"]:
            return
        hook_flags["backbone"] = True
        out_shapes = _shape(out if not isinstance(out, dict) else list(out.values()))
        # region agent log
        dlog(
            run_id,
            "H3",
            "adapter_smoke_check.py:108",
            "img_backbone_forward_seen",
            {"output_shapes": out_shapes},
        )
        # endregion

    def _transformer_hook(_module, _inp, out):
        if hook_flags["transformer"]:
            return
        hook_flags["transformer"] = True
        bev_embed_shape = None
        try:
            if isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
                bev_embed_shape = list(out[0].shape)
        except Exception:
            bev_embed_shape = None
        # region agent log
        dlog(
            run_id,
            "H4",
            "adapter_smoke_check.py:128",
            "transformer_forward_seen",
            {"bev_embed_shape": bev_embed_shape, "output_shape_tree": _shape(out)},
        )
        # endregion

    h1 = model.img_backbone.register_forward_hook(_backbone_hook)
    h2 = model.pts_bbox_head.transformer.register_forward_hook(_transformer_hook)

    data_batch = next(iter(runner.train_dataloader))
    processed = model.data_preprocessor(data_batch, training=True)
    imgs = processed["inputs"].get("imgs", processed["inputs"].get("img", None))
    img_shape = None
    img_dtype = None
    img_mean = None
    img_std = None
    img_container_type = str(type(imgs))
    if torch.is_tensor(imgs):
        img_shape = list(imgs.shape)
        img_dtype = str(imgs.dtype)
        img_mean = float(imgs.float().mean().item())
        img_std = float(imgs.float().std().item())
    elif isinstance(imgs, (list, tuple)) and len(imgs) > 0 and torch.is_tensor(imgs[0]):
        img_shape = [list(x.shape) if torch.is_tensor(x) else str(type(x)) for x in imgs]
        img_dtype = str(imgs[0].dtype)
        img_mean = float(imgs[0].float().mean().item())
        img_std = float(imgs[0].float().std().item())
    # region agent log
    dlog(
        run_id,
        "H2",
        "adapter_smoke_check.py:144",
        "batch_processed",
        {
            "img_container_type": img_container_type,
            "img_shape": img_shape,
            "img_dtype": img_dtype,
            "img_mean": img_mean,
            "img_std": img_std,
        },
    )
    # endregion

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    losses = model.loss(processed["inputs"], processed["data_samples"])
    loss = sum(v for v in losses.values() if torch.is_tensor(v))
    peak_after_fwd = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    did_backward = False
    if loss.requires_grad:
        loss.backward()
        did_backward = True

    post_trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    post_with_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]

    print("Trainable params after first forward:", len(post_trainable))
    for n in post_trainable[:60]:
        print("  ", n)

    print("Params with gradients:", len(post_with_grad))
    for n in post_with_grad[:60]:
        print("  ", n)

    bad_prefixes = ("img_backbone", "img_neck", "pts_bbox_head", "depth_adapter.depth_backbone")
    bad_trainable = [n for n in post_trainable if n.startswith(bad_prefixes)]
    print("Unexpected trainable core params:", len(bad_trainable))
    for n in bad_trainable[:20]:
        print("  ", n)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"Peak CUDA memory allocated: {peak_gb:.3f} GB")
    print(f"Peak CUDA memory after forward: {peak_after_fwd:.3f} GB")
    print(f"Backward executed: {did_backward}")
    # region agent log
    dlog(
        run_id,
        "H5",
        "adapter_smoke_check.py:191",
        "post_backward_stats",
        {
            "trainable_count": len(post_trainable),
            "grad_count": len(post_with_grad),
            "unexpected_core_trainable": len(bad_trainable),
            "peak_cuda_gb": float(peak_gb),
            "peak_after_fwd_gb": float(peak_after_fwd),
            "did_backward": bool(did_backward),
            "backbone_hook_seen": bool(hook_flags["backbone"]),
            "transformer_hook_seen": bool(hook_flags["transformer"]),
        },
    )
    # endregion
    h1.remove()
    h2.remove()


if __name__ == "__main__":
    main()
