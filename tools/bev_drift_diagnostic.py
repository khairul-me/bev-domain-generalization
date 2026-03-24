import copy
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner


def measure_drift(feat_a, feat_b):
    a = feat_a.flatten(1)
    b = feat_b.flatten(1)
    return F.cosine_similarity(a, b, dim=1).mean().item()


def main():
    # Load config with adapter DISABLED
    cfg = Config.fromfile("E:/bev_research/configs/bevformer_rtx5060.py")
    cfg.model.pop("depth_adapter", None)
    cfg.model.freeze_bevformer = False
    cfg.custom_hooks = []
    cfg.work_dir = "E:/bev_research/experiments/diagnostics/bev_drift"

    # Build runner/model and load pretrained checkpoint from cfg.load_from
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model
    model.eval()
    model.cuda()

    # Storage for captured features
    captured = {}

    def hook_img_feats(module, input, output):
        if isinstance(output, (list, tuple)):
            captured["img_feats"] = [f.detach().cpu() for f in output]
        else:
            captured["img_feats"] = [output.detach().cpu()]

    def hook_bev_embed(module, input, output):
        # output is bev_embed from encoder
        if isinstance(output, (list, tuple)):
            captured["bev_embed"] = output[0].detach().cpu()
        else:
            captured["bev_embed"] = output.detach().cpu()

    # Attach hooks
    h_img = model.img_neck.register_forward_hook(hook_img_feats)
    h_bev = model.pts_bbox_head.transformer.encoder.register_forward_hook(hook_bev_embed)

    # Confirm split PKLs exist and load first 20 infos (for reporting only)
    with open("C:/datasets/nuscenes/nuscenes_infos_temporal_val_boston.pkl", "rb") as f:
        boston_infos = pickle.load(f)["infos"][:20]
    with open("C:/datasets/nuscenes/nuscenes_infos_temporal_val_singapore.pkl", "rb") as f:
        singapore_infos = pickle.load(f)["infos"][:20]
    pair_count = min(len(boston_infos), len(singapore_infos), 20)

    # Build Boston/Singapore dataloaders using split ann files
    boston_loader_cfg = copy.deepcopy(cfg.val_dataloader)
    boston_loader_cfg["dataset"]["ann_file"] = "C:/datasets/nuscenes/nuscenes_infos_temporal_val_boston.pkl"
    singapore_loader_cfg = copy.deepcopy(cfg.val_dataloader)
    singapore_loader_cfg["dataset"]["ann_file"] = "C:/datasets/nuscenes/nuscenes_infos_temporal_val_singapore.pkl"
    boston_loader = Runner.build_dataloader(boston_loader_cfg)
    singapore_loader = Runner.build_dataloader(singapore_loader_cfg)

    boston_iter = iter(boston_loader)
    singapore_iter = iter(singapore_loader)

    img_feat_drifts = []
    bev_embed_drifts = []

    with torch.no_grad():
        for i in range(pair_count):
            # Run Boston frame i
            captured.clear()
            batch_boston = next(boston_iter)
            _ = model.test_step(batch_boston)
            boston_img = [f.clone() for f in captured["img_feats"]]
            boston_bev = captured["bev_embed"].clone()

            # Run Singapore frame i
            captured.clear()
            batch_singapore = next(singapore_iter)
            _ = model.test_step(batch_singapore)
            sing_img = [f.clone() for f in captured["img_feats"]]
            sing_bev = captured["bev_embed"].clone()

            # Measure drift
            img_drift = np.mean([measure_drift(boston_img[l], sing_img[l]) for l in range(len(boston_img))])
            bev_drift = measure_drift(boston_bev, sing_bev)

            img_feat_drifts.append(img_drift)
            bev_embed_drifts.append(bev_drift)
            print(f"Pair {i:02d} — img_feat cosine: {img_drift:.4f} | bev_embed cosine: {bev_drift:.4f}")

    print(f"\nMean img_feat drift:  {np.mean(img_feat_drifts):.4f}")
    print(f"Mean bev_embed drift: {np.mean(bev_embed_drifts):.4f}")

    h_img.remove()
    h_bev.remove()


if __name__ == "__main__":
    main()

