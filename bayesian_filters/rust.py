"""Experimental, opt-in Rust kernels; the existing NumPy APIs remain the reference.

Build the separate extension with the instructions in scratch_files/RUST_BACKEND.md.
These functions do not globally replace or monkeypatch existing filter classes.
"""

import numpy as np

try:
    import _bayesian_filters_rust as _native
except ImportError as exc:
    raise ImportError("Build/install the optional Rust extension first; see scratch_files/RUST_BACKEND.md") from exc


def _array(value, ndim):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != ndim:
        raise ValueError(f"expected a {ndim}-dimensional array, got {out.ndim}")
    return out


def kalman_step(x, P, z, F, Q, H, R):
    """Predict then update; constant matrices, flat state, no control or missing z.

    Returns (posterior state, posterior covariance), using the Joseph update.
    Singular innovation covariance raises ValueError. Inputs are not mutated.
    """
    return _native.kalman_step(
        _array(x, 1),
        _array(P, 2),
        _array(z, 1),
        _array(F, 2),
        _array(Q, 2),
        _array(H, 2),
        _array(R, 2),
    )


def kalman_batch(x, P, zs, F, Q, H, R):
    """Fused predict/update sequence; return posterior and prior means/covariances.

    Means have shape (steps, state_dim); covariances (steps, state_dim, state_dim).
    This restricted API requires fixed F/Q/H/R, flat x, and finite measurements.
    It does not mutate an existing KalmanFilter or implement its optional hooks.
    """
    return _native.kalman_batch(
        _array(x, 1),
        _array(P, 2),
        _array(zs, 2),
        _array(F, 2),
        _array(Q, 2),
        _array(H, 2),
        _array(R, 2),
    )


def unscented_transform(sigmas, Wm, Wc, noise_cov=None, mean_fn=None, residual_fn=None):
    """Rust weighted moments, or the reference path for custom Python callbacks.

    Can be passed as UT to existing UKF predict/update calls. Python fx/hx and
    sigma-point construction still execute through the existing implementation.
    """
    if mean_fn is not None or (residual_fn is not None and residual_fn is not np.subtract):
        from .kalman.unscented_transform import unscented_transform as reference

        return reference(sigmas, Wm, Wc, noise_cov, mean_fn, residual_fn)
    sigmas = _array(sigmas, 2)
    noise = np.zeros((sigmas.shape[1], sigmas.shape[1])) if noise_cov is None else _array(noise_cov, 2)
    return _native.unscented_transform(sigmas, _array(Wm, 1), _array(Wc, 1), noise)


def gh_batch(data, x, dx, dt, g, h, save_predictions=False):
    """Scalar GHFilter.batch_filter equivalent, without Saver callbacks."""
    results, predictions = _native.gh_batch(_array(data, 1), x, dx, dt, g, h)
    return (results, predictions) if save_predictions else results


def discrete_update(likelihood, prior):
    """Return normalized nonnegative posterior; reject zero/nonfinite total mass."""
    return _native.discrete_update(_array(likelihood, 1), _array(prior, 1))


def discrete_predict(pdf, offset, kernel):
    """Integer-offset, one-dimensional wrap-mode discrete Bayes prediction."""
    if not isinstance(offset, (int, np.integer)):
        raise ValueError("offset must be an integer")
    return _native.discrete_predict(_array(pdf, 1), int(offset), _array(kernel, 1))


def systematic_resample(weights):
    """Use NumPy's existing RNG stream; perform the cumulative traversal in Rust."""
    weights = _array(weights, 1)
    n = len(weights)
    if not n:
        raise ValueError("weights must not be empty")
    positions = (np.random.random() + np.arange(n)) / n
    return _native.resample(weights, positions)


def stratified_resample(weights):
    """Use NumPy RNG with native traversal; output index dtype is int64."""
    weights = _array(weights, 1)
    n = len(weights)
    if not n:
        raise ValueError("weights must not be empty")
    positions = (np.random.random(n) + np.arange(n)) / n
    return _native.resample(weights, positions)
