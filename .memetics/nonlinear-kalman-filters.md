---
upstream:
  - repo: rlabbe/filterpy
    paths:
      - filterpy/kalman/EKF.py
      - filterpy/kalman/UKF.py
      - filterpy/kalman/sigma_points.py
      - filterpy/kalman/unscented_transform.py
      - filterpy/kalman/CubatureKalmanFilter.py
      - filterpy/kalman/ensemble_kalman_filter.py
    ref: master
    last_seen: e164fc1963f95d0eada108b78a4422bfac5ce8b1
local:
  paths:
    - bayesian_filters/kalman/EKF.py
    - bayesian_filters/kalman/UKF.py
    - bayesian_filters/kalman/sigma_points.py
    - bayesian_filters/kalman/unscented_transform.py
    - bayesian_filters/kalman/CubatureKalmanFilter.py
    - bayesian_filters/kalman/ensemble_kalman_filter.py
---

# Nonlinear Kalman filters (EKF, UKF + sigma points, CKF, EnKF)

The nonlinear estimators are ported one-to-one from filterpy: `ExtendedKalmanFilter` (with
`predict_update`), `UnscentedKalmanFilter` with the `unscented_transform` function and the
sigma-point generators (`MerweScaledSigmaPoints`, `JulierSigmaPoints`, `SimplexSigmaPoints`),
the `CubatureKalmanFilter` with `spherical_radial_sigmas` / `ckf_transform`, and the
`EnsembleKalmanFilter`. These are the most numerically delicate modules in the package
(sigma-point weights, square-root choice, residual/mean functions for angular state), so upstream
changes here are worth diffing carefully.

Adaptation notes (keep stable when syncing from upstream):

- `filterpy.*` imports are rewritten to `bayesian_filters.*`; ruff formatting applies (long
  `__init__` signatures are exploded one argument per line).
- Keep the pluggable hooks exactly as upstream exposes them: `sqrt_fn`, `x_mean_fn`, `z_mean_fn`,
  `residual_x`, `residual_z`, `state_add` on the UKF, and `HJacobian`/`Hx` callables on the EKF.
  User code written against filterpy must keep working after an import rename.
- `sigma_points.py` must keep supporting all three sigma-point families with the same `num_sigmas()`
  / `sigma_points()` / `Wm` / `Wc` contract, because the UKF only depends on that contract.
- Tests live in `bayesian_filters/kalman/tests/` (`test_ukf.py`, `test_ekf.py`, `test_ckf.py`,
  `test_enkf.py`, `test_sigma_points.py`, `test_unscented_transform.py`); port upstream fixes with a test.
