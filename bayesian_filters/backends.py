"""Explicit selection for the optional kernel API; no global backend mutation."""

from types import SimpleNamespace


def get_backend(name="numpy"):
    """Return the NumPy reference or optional Rust implementation of shared kernels.

    This is a limited common API, not a replacement for every filter class.
    Importing the default backend does not require the native extension.
    """
    if name == "rust":
        from . import rust

        return rust
    if name != "numpy":
        raise ValueError("backend must be 'numpy' or 'rust'")
    from .kalman import KalmanFilter, unscented_transform
    from .gh import GHFilter
    from .discrete_bayes import predict, update
    from .monte_carlo import systematic_resample, stratified_resample

    def configure(x, P, F, Q, H, R):
        import numpy as np

        x = np.asarray(x, dtype=float)
        H = np.asarray(H, dtype=float)
        k = KalmanFilter(len(x), H.shape[0])
        k.x = x.copy()
        k.P = np.asarray(P, dtype=float).copy()
        k.F, k.Q, k.H, k.R = (np.asarray(a, dtype=float) for a in (F, Q, H, R))
        return k

    def kalman_step(x, P, z, F, Q, H, R):
        k = configure(x, P, F, Q, H, R)
        k.predict()
        k.update(z)
        return k.x, k.P

    def kalman_batch(x, P, zs, F, Q, H, R):
        return configure(x, P, F, Q, H, R).batch_filter(zs)

    def gh_batch(data, x, dx, dt, g, h, save_predictions=False):
        return GHFilter(x, dx, dt, g, h).batch_filter(data, save_predictions=save_predictions)

    return SimpleNamespace(
        kalman_step=kalman_step,
        kalman_batch=kalman_batch,
        unscented_transform=unscented_transform,
        gh_batch=gh_batch,
        discrete_update=update,
        discrete_predict=predict,
        systematic_resample=systematic_resample,
        stratified_resample=stratified_resample,
    )
