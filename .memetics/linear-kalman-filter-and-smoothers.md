---
upstream:
  - repo: rlabbe/filterpy
    paths:
      - filterpy/kalman/kalman_filter.py
      - filterpy/kalman/fixed_lag_smoother.py
      - filterpy/kalman/information_filter.py
      - filterpy/kalman/square_root.py
      - filterpy/kalman/fading_memory.py
    ref: master
    last_seen: b83ddad98ef09b2a60cd5d90dd379fedd8776d16
local:
  paths:
    - bayesian_filters/kalman/kalman_filter.py
    - bayesian_filters/kalman/fixed_lag_smoother.py
    - bayesian_filters/kalman/information_filter.py
    - bayesian_filters/kalman/square_root.py
    - bayesian_filters/kalman/fading_memory.py
---

# Linear Kalman filter family (KF, RTS, fixed-lag, information, square-root, fading-memory)

The linear Kalman filter core of this package is a direct port of filterpy's `kalman` package:
the object-oriented `KalmanFilter` plus the procedural `predict`/`update`/`batch_filter` functions,
the Rauch-Tung-Striebel smoother (`rts_smoother`), the `FixedLagSmoother`, the `InformationFilter`,
the `SquareRootKalmanFilter` and the `FadingKalmanFilter`. filterpy was forked rather than imported
because upstream stopped merging packaging fixes (Python 3.12+, setuptools deprecations); the
algorithms themselves are unchanged and this fork tracks upstream so bug fixes can still be pulled in.

Adaptation notes (keep stable when syncing from upstream):

- Package is renamed `filterpy.*` -> `bayesian_filters.*` everywhere (imports, docstrings, doc URLs).
  Upstream import lines like `from filterpy.common import ...` must be rewritten, never copied.
- Code is formatted with ruff (`ruff format`, double quotes, long signatures exploded one arg per
  line). Reformatting noise in a diff is not a semantic change; only port logic/bug fixes.
- Public API surface (class names, attribute names `x, P, Q, F, H, R, K, y, S, z`, `log_likelihood`,
  `likelihood`, `mahalanobis`, `Saver` support, `alpha` fading memory) must stay identical to upstream
  so filterpy users can switch by changing the import.
- Each module has a dedicated pytest file under `bayesian_filters/kalman/tests/`; new upstream
  behaviour should come with a test there rather than relying on upstream's doctests.
