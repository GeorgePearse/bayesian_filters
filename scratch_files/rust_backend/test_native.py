"""Differential tests against the unchanged NumPy/SciPy implementation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from bayesian_filters import rust
from bayesian_filters.kalman import KalmanFilter, unscented_transform
from bayesian_filters.gh import GHFilter
from bayesian_filters.discrete_bayes import predict, update
from bayesian_filters import monte_carlo


@pytest.mark.parametrize("n,m", [(1, 1), (2, 1), (4, 2), (9, 3), (16, 8), (64, 16), (128, 32)])
def test_kalman(n, m):
    rng = np.random.default_rng(n)
    a = rng.normal(size=(n, n))
    p = a @ a.T + np.eye(n)
    f = np.eye(n) * 0.9
    h = rng.normal(size=(m, n))
    q = np.eye(n) * 0.01
    r = np.eye(m) * 0.1
    x = rng.normal(size=n)
    zs = rng.normal(size=(25, m))
    k = KalmanFilter(n, m)
    k.x = x.copy()
    k.P = p.copy()
    k.F = f
    k.H = h
    k.Q = q
    k.R = r
    expected = k.batch_filter(zs)
    actual = rust.kalman_batch(x, p, zs, f, q, h, r)
    for e, v in zip(expected, actual):
        assert_allclose(v, e, rtol=2e-10, atol=2e-10)
    assert_allclose(rust.kalman_step(x, p, zs[0], f, q, h, r)[0], expected[0][0], rtol=2e-10, atol=2e-10)
    # Read-only/noncontiguous input is accepted without aliasing or mutation.
    zs.setflags(write=False)
    for e, v in zip(rust.kalman_batch(x, p, zs[::2], f, q, h, r), rust.kalman_batch(x, p, zs[::2].copy(), f, q, h, r)):
        assert_allclose(e, v)


@pytest.mark.parametrize("n", [1, 2, 6, 16, 32, 128])
def test_transform(n):
    rng = np.random.default_rng(n)
    s = rng.normal(size=(2 * n + 1, n))
    w = rng.normal(size=2 * n + 1)
    w /= w.sum()
    wc = w.copy()
    wc[0] -= 2
    for noise in [None, np.eye(n) * 0.1]:
        for a, b in zip(rust.unscented_transform(s, w, wc, noise), unscented_transform(s, w, wc, noise)):
            assert_allclose(a, b, rtol=1e-10, atol=1e-10)

    def mean(s, w):
        return np.dot(w, s)

    def residual(a, b):
        return a - b

    for a, b in zip(
        rust.unscented_transform(s, w, wc, mean_fn=mean, residual_fn=residual),
        unscented_transform(s, w, wc, mean_fn=mean, residual_fn=residual),
    ):
        assert_allclose(a, b)


def test_gh():
    z = np.random.default_rng(0).normal(size=100)
    ref = GHFilter(x=1.0, dx=0.2, dt=0.1, g=0.3, h=0.01)
    for a, b in zip(rust.gh_batch(z, 1.0, 0.2, 0.1, 0.3, 0.01, True), ref.batch_filter(z, save_predictions=True)):
        assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    assert rust.gh_batch([], 1.0, 0.2, 0.1, 0.3, 0.01).shape == (1, 2)


@pytest.mark.parametrize("offset", [-100, -3, 0, 2, 100])
@pytest.mark.parametrize("length", [1, 2, 3, 6])
def test_discrete(offset, length):
    rng = np.random.default_rng(3)
    p = rng.random(17)
    p /= p.sum()
    k = rng.random(length)
    k /= k.sum()
    assert_allclose(rust.discrete_predict(p, offset, k), predict(p, offset, k), rtol=1e-12, atol=1e-12)
    assert_allclose(rust.discrete_update(k, k), update(k, k), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["systematic_resample", "stratified_resample"])
@pytest.mark.parametrize("n", [1, 10, 10000])
def test_resample(kind, n):
    w = np.random.default_rng(0).random(n)
    w /= w.sum()
    np.random.seed(42)
    expected = getattr(monte_carlo, kind)(w)
    np.random.seed(42)
    actual = getattr(rust, kind)(w)
    assert_array_equal(expected, actual)


def test_rejected_inputs():
    with pytest.raises(ValueError):
        rust.discrete_update([0.0], [1.0])
    with pytest.raises(ValueError):
        rust.discrete_update([np.nan], [1.0])
    with pytest.raises(ValueError):
        rust.systematic_resample([0.1, 0.1])
    with pytest.raises(ValueError):
        rust.gh_batch([1.0], 0.0, 0.0, 0.0, 0.1, 0.1)
    with pytest.raises(ValueError):
        rust.kalman_step([0.0], [[0.0]], [0.0], [[1.0]], [[0.0]], [[1.0]], [[0.0]])
    with pytest.raises(ValueError):
        rust.kalman_step([0.0, 0.0], [[1.0]], [0.0], [[1.0]], [[0.0]], [[1.0]], [[1.0]])
    with pytest.raises(ValueError):
        rust.unscented_transform(np.zeros((3, 2)), [1.0], [1.0], np.eye(2))


@pytest.mark.parametrize("n", [4, 5, 131072, 131073])
def test_simd_and_parallel_update(n):
    rng = np.random.default_rng(0)
    a = rng.random(n)
    b = rng.random(n)
    assert_allclose(rust.discrete_update(a, b), update(a, b), rtol=1e-12, atol=1e-12)
    assert_allclose(rust.discrete_update(a[::-1], b[::-1]), update(a[::-1], b[::-1]), rtol=1e-12, atol=1e-12)
    a[-1] = np.inf
    with pytest.raises(ValueError):
        rust.discrete_update(a, b)


def test_backend_selection_does_not_replace_reference():
    from bayesian_filters.backends import get_backend
    from bayesian_filters.kalman import unscented_transform as before

    native = get_backend("rust")
    reference = get_backend()
    assert native.unscented_transform is rust.unscented_transform
    assert reference.unscented_transform is before
    assert_allclose(
        native.discrete_update([0.2, 0.8], [0.5, 0.5]),
        reference.discrete_update(np.array([0.2, 0.8]), np.array([0.5, 0.5])),
    )
    with pytest.raises(ValueError):
        get_backend("unknown")


@pytest.mark.parametrize("length", [2, 3, 6])
def test_parallel_convolution(length):
    rng = np.random.default_rng(6)
    p = rng.random(131073)
    p /= p.sum()
    k = rng.random(length)
    k /= k.sum()
    for offset in [-200000, 0, 200000]:
        assert_allclose(rust.discrete_predict(p, offset, k), predict(p, offset, k), rtol=1e-12, atol=1e-12)
        assert_allclose(rust.discrete_predict(p[::-1], offset, k), predict(p[::-1], offset, k), rtol=1e-12, atol=1e-12)


def test_subnormal_mass_and_noncontiguous_transform():
    likelihood = np.full(4, np.nextafter(0.0, 1.0))
    prior = np.ones(4)
    assert_allclose(rust.discrete_update(likelihood, prior), update(likelihood, prior))
    sigmas = np.arange(15.0).reshape(5, 3)[::-1, ::-1]
    weights = np.full(5, 0.2)
    for actual, expected in zip(
        rust.unscented_transform(sigmas, weights, weights), unscented_transform(sigmas, weights, weights)
    ):
        assert_allclose(actual, expected)
