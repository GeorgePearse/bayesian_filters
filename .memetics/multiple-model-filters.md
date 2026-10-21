---
upstream:
  - repo: rlabbe/filterpy
    paths:
      - filterpy/kalman/IMM.py
      - filterpy/kalman/mmae.py
    ref: master
    last_seen: 6e6071d73f9eecf6bbca7f813895136e443c4278
local:
  paths:
    - bayesian_filters/kalman/IMM.py
    - bayesian_filters/kalman/mmae.py
---

# Multiple-model estimators (IMM, MMAE)

`IMMEstimator` (interacting multiple model) and `MMAEFilterBank` (multiple model adaptive
estimation) are ported from filterpy. Both wrap a bank of existing filters and mix or weight their
estimates by likelihood, so they depend on the `likelihood` attribute and the `x`/`P` layout of
the filters in this package; a change in either upstream file usually implies a change to that
contract.

Adaptation notes (keep stable when syncing from upstream):

- `filterpy.*` imports are rewritten to `bayesian_filters.*`; ruff formatting applies.
- Keep the constructor contract: `IMMEstimator(filters, mu, M)` with mode probabilities `mu`
  and Markov transition matrix `M`; `MMAEFilterBank(filters, p, dim_x, H=None)`. Mixed state is
  exposed as `x`/`P` plus the per-model `x_prior`/`P_prior`/`x_post`/`P_post` as upstream does.
- The bank accepts any filter object with `predict()`, `update()`, `x`, `P` and `likelihood`,
  so it must not be tied to `KalmanFilter` specifically.
- Tests: `bayesian_filters/kalman/tests/test_imm.py`, `test_mmae.py`.
