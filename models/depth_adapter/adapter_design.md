# Adapter Design (Gate D2)

## Scope

Design a lightweight depth-prior adapter for BEVFormer domain generalization on
`nuScenes-Boston -> nuScenes-Singapore`, using a frozen DAv2 backbone.

## Locked Design Decisions

1. **Inject encoder features, not final depth output**
   - Use frozen DAv2/DINOv2 intermediate encoder features.
   - Do **not** inject final scalar depth map values.

2. **Residual fusion only**
   - Adapter output is added to BEVFormer image features:
     - `fused_feat = bevformer_feat + alpha * adapter(depth_feat)`
   - Adapter never replaces BEVFormer features.

3. **Injection point: before BEV spatial cross-attention**
   - Inject into multi-level image features in `extract_img_feat()`.
   - This ensures geometric priors affect BEV query image sampling.

4. **Source-domain-only training with frozen backbones**
   - Train on Boston/source split only.
   - Freeze DAv2 encoder parameters.
   - Freeze BEVFormer parameters.
   - Keep adapter trainable.

## Tensor-Level Specification

- BEVFormer image features per level: `F_l in R^{BN x C_l x H_l x W_l}`
- DAv2 encoder feature: `D in R^{BN x C_d x H_d x W_d}`
- For each level `l`:
  - Resize: `D_l = interpolate(D, size=(H_l, W_l))`
  - Project: `A_l = Conv1x1-ReLU-Conv1x1(D_l) in R^{BN x C_l x H_l x W_l}`
  - Residual fuse: `F'_l = F_l + alpha * A_l`

Where:
- `BN = batch_size * num_cameras`
- `C_d = 384` for DAv2 ViT-S
- `alpha` is configurable residual scaling (default `1.0`).

## Implementation Notes

- DAv2 input reconstruction:
  - Denormalize BEVFormer camera tensor using `img_norm_cfg`.
  - Convert to RGB [0,1], then ImageNet normalize for DAv2.
- DAv2 encoder forward uses `torch.no_grad()`.
- Adapter modules are per-FPN-level 1x1 projection MLPs.

## Failure Handling and Caveat

- DAv2 may degrade on extreme low-light/night edge cases.
- Residual fusion guarantees BEVFormer baseline path remains intact when depth
  quality is poor.

## Verification Checklist

- [x] DAv2 features are taken from frozen encoder intermediate layers.
- [x] Fusion is residual addition, not replacement.
- [x] Injection occurs before transformer spatial cross-attention.
- [x] Freeze mode exists to keep BEVFormer + DAv2 frozen.
- [ ] Run a 1-iter smoke test with adapter enabled.
- [ ] Confirm only adapter parameters have gradients.
