## Smoke Test Result

- Date: 2026-03-15
- Config: `E:/bev_research/configs/bevformer_rtx5060.py`
- Checkpoint: `E:/bev_research/checkpoints/bevformer_base_epoch_24.pth`
- Run log: `E:/bev_research/logs/smoke_test_real_loss_v4.log`

### Verified Loss Metrics

- Iteration 10 (`Epoch(train) [1][10/28130]`):
  - `loss`: 16.6742
  - `loss_cls`: 1.3847
  - `loss_bbox`: 1.4565
- Iteration 20 (`Epoch(train) [1][20/28130]`):
  - `loss`: 12.0746
  - `loss_cls`: 0.7991
  - `loss_bbox`: 1.2261

### Runtime Metrics

- Peak VRAM observed (`memory`): 4748 MB
- Iteration speed:
  - Iter 10 window: 6.5724 s/iter
  - Iter 20 window: 5.7959 s/iter
  - Approx mean: 6.1842 s/iter

### Validation Notes

- Decoder shape mismatch resolved (no RuntimeError in decoder forward).
- Loss values are real and non-zero.
- Loss changed meaningfully between logs (downward trend from iter 10 to 20).
- `forward_pts_train` fallback is removed and real loss path executes.
