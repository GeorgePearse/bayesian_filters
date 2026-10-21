---
upstream:
  - repo: rlabbe/filterpy
    paths:
      - filterpy/gh/gh_filter.py
      - filterpy/discrete_bayes/discrete_bayes.py
      - filterpy/leastsq/least_squares.py
      - filterpy/hinfinity/hinfinity_filter.py
      - filterpy/memory/fading_memory.py
      - filterpy/common/discretization.py
      - filterpy/common/kinematic.py
      - filterpy/common/helpers.py
      - filterpy/stats/stats.py
    ref: master
    last_seen: 3d327f645642f2310b16ad0f4f31bb6f3a112483
local:
  paths:
    - bayesian_filters/gh/gh_filter.py
    - bayesian_filters/discrete_bayes/discrete_bayes.py
    - bayesian_filters/leastsq/least_squares.py
    - bayesian_filters/hinfinity/hinfinity_filter.py
    - bayesian_filters/memory/fading_memory.py
    - bayesian_filters/common/discretization.py
    - bayesian_filters/common/kinematic.py
    - bayesian_filters/common/helpers.py
    - bayesian_filters/stats/stats.py
---

# Classical estimators and the shared noise/statistics helpers

The non-Kalman estimators from filterpy are ported as separate subpackages, mirroring upstream's
layout: the g-h filters (`GHFilter`, `GHKFilter`, `GHFilterOrder`, `least_squares_parameters`,
`optimal_noise_smoothing`), the discrete Bayes histogram filter (`predict`/`update`/`normalize`),
the recursive `LeastSquaresFilter`, the `HInfinityFilter` and the `FadingMemoryFilter`. Alongside
them sit the helpers every filter in the package relies on: process-noise discretisation
(`Q_discrete_white_noise`, `Q_continuous_white_noise`, `van_loan_discretization`),
kinematic state-transition builders (`kinematic_kf`, `kinematic_state_transition`), `Saver`,
`pretty_str`, `reshape_z`, `inv_diagonal`, and the statistics module (`logpdf`, `gaussian`,
`multivariate_gaussian`, `mahalanobis`, covariance-ellipse plotting). They are kept in the
fork because the Kalman modules import them and users expect them under the same names.

Adaptation notes (keep stable when syncing from upstream):

- `filterpy.*` imports are rewritten to `bayesian_filters.*`; ruff formatting applies.
- `common.helpers` and `stats.stats` are imported by nearly every filter module, so a signature
  change there must be propagated across the package in the same change (e.g. `logpdf`,
  `pretty_str`, `reshape_z`, `Saver`).
- Keep the sub-package names (`gh`, `discrete_bayes`, `leastsq`, `hinfinity`, `memory`) as
  upstream has them so the import rename is the only migration step for filterpy users.
- Each sub-package has its own `tests/` directory with pytest files; port upstream fixes with tests.
