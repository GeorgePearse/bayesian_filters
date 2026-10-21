---
upstream:
  - repo: rlabbe/filterpy
    paths:
      - filterpy/monte_carlo/resampling.py
    ref: master
    last_seen: 6cc052bea5a806d3531b59c820530c42b67a910c
local:
  paths:
    - bayesian_filters/monte_carlo/resampling.py
---

# Particle filter resampling schemes

The four particle-filter resampling algorithms (`residual_resample`, `stratified_resample`,
`systematic_resample`, `multinomial_resample`) are ported from filterpy's `monte_carlo` package.
The functions are tiny but subtle (the `cumulative_sum[-1] = 1.0` guard against round-off, the
exact stratification offsets), so upstream fixes to any of them are worth pulling in verbatim.

Adaptation notes (keep stable when syncing from upstream):

- Every function takes a normalised 1-D `weights` array and returns an integer index array of
  length `N`; that signature is the whole public API and must not change.
- Formatting is ruff style (double quotes, spaced operators); diffs against upstream are mostly
  that noise plus the `bayesian_filters` URLs in the module docstring.
- Tests: `bayesian_filters/monte_carlo/tests/test_resampling.py`.
