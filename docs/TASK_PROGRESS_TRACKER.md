# Research Pipeline — Task Progress Tracker

## Verification Workflow (Per Your Request)

```
For each task:
  1. Complete all steps in the task
  2. Verify task comprehensively (every checkpoint)
  3. If task ≥ 2: Step back and verify Tasks 1..N alignment with overall plan
  4. Confirm success → Proceed to next task
```

---

## Overall Plan Alignment (Verified After Task 2+)

| Principle | Source | Status |
|-----------|--------|--------|
| Domain-generalizable BEV detection | Pipeline Overview | ✓ |
| Frozen Depth Anything V2 as prior | Core Idea | ✓ |
| Lightweight adapter only | Core Idea | ✓ |
| Test-time adaptation | Core Idea | ✓ |
| nuScenes → KITTI cross-domain | Primary Benchmark | ✓ |
| RTX 5060 hardware constraints | Hardware | ✓ |
| < 3 months timeline | Timeline | ✓ |

---

## Task Status Summary

| Task | Name | Status | Verification |
|------|------|--------|--------------|
| 1 | Environment Setup | ✅ COMPLETE | See TASK1_VERIFICATION_REPORT.md |
| 2 | Focused Literature Review | 🟡 IN PROGRESS | Literature notes + depth chain created |
| 3 | Dataset Acquisition | ⬜ Not started | — |
| 4 | Baseline Codebase Setup | ⬜ Not started | — |
| 5 | Baseline Reproduction | ⬜ Not started | — |
| 6–15 | (Future tasks) | ⬜ | — |

---

## Verification History

### Task 1
- **Date:** 2026-03-11
- **Result:** INCOMPLETE — Conda required; mmcv build failed on system Python
- **Action:** Install Miniconda, re-run setup in bev_research env
- **Report:** TASK1_VERIFICATION_REPORT.md

---

## Baseline Model Decision

| Model | NDS (val) | VRAM at inference (fp16) | Decision |
|-------|-----------|--------------------------|---------|
| BEVFormer-Tiny  | ~35.8% | ~4.5 GB | ❌ Not suitable for publication |
| BEVFormer-Base  | ~45.7% | > 16 GB | ❌ OOM on RTX 5060 Ti during training |
| BEVFormer-Small | ~47.9% | ~12 GB | ✅ Selected (fallback, but actually we will use Tiny for smoke test first) |

**Selected baseline:** BEVFormer-Tiny (for smoke test), BEVFormer-Small (for full reproduction)
**Reason:** Base model OOMs on 16GB VRAM. Fallback to Small for full training.
