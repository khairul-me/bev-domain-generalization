import torch
from mmengine.config import Config
from mmengine.runner import Runner


def main():
    cfg = Config.fromfile("E:/bev_research/configs/bevformer_rtx5060.py")
    cfg.work_dir = "E:/bev_research/experiments/E3_adapter/overfit_check"
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.train_dataloader.dataset.queue_length = 4

    runner = Runner.from_cfg(cfg)
    model = runner.model
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable_params={sum(p.numel() for p in trainable)}")
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)

    batch = next(iter(runner.train_dataloader))
    processed = model.data_preprocessor(batch, training=True)

    losses = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, 31):
        optimizer.zero_grad(set_to_none=True)
        out = model.loss(processed["inputs"], processed["data_samples"])
        loss = sum(v for v in out.values() if torch.is_tensor(v))
        loss.backward()
        optimizer.step()
        loss_val = float(loss.detach().cpu())
        losses.append(loss_val)
        if step % 5 == 0:
            print(f"step={step} loss={loss_val:.4f}")

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"loss_start={losses[0]:.4f} loss_end={losses[-1]:.4f}")
    print(f"loss_min={min(losses):.4f} loss_max={max(losses):.4f}")
    print(f"peak_cuda_gb={peak_gb:.3f}")


if __name__ == "__main__":
    main()
