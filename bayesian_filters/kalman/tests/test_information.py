# -*- coding: utf-8 -*-
"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
https://github.com/GeorgePearse/bayesian_filters

Documentation at:
https://georgepearse.github.io/bayesian_filters

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy.random as random
import numpy as np
import matplotlib.pyplot as plt

from bayesian_filters.common import Saver
from bayesian_filters.kalman import KalmanFilter, InformationFilter


DO_PLOT = False


def test_1d_0P():
    global inf
    f = KalmanFilter(dim_x=2, dim_z=1)
    inf = InformationFilter(dim_x=2, dim_z=1)

    f.x = np.array([[2.0], [0.0]])  # initial state (location and velocity)

    f.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # state transition matrix

    f.H = np.array([[1.0, 0.0]])  # Measurement function
    f.R = np.array([[5.0]])  # state uncertainty
    f.Q = np.eye(2) * 0.0001  # process uncertainty
    f.P = np.diag([20.0, 20.0])

    inf.x = f.x.copy()
    inf.F = f.F.copy()
    inf.H = np.array([[1.0, 0.0]])  # Measurement function
    inf.R_inv *= 1.0 / 5  # state uncertainty
    inf.Q = np.eye(2) * 0.0001
    inf.P_inv = 0.000000000000000000001
    # inf.P_inv = inv(f.P)

    m = []
    r = []
    r2 = []

    zs = []
    for t in range(50):
        # create measurement = t plus white noise
        z = t + random.randn() * np.sqrt(5)
        zs.append(z)

        # perform kalman filtering
        f.predict()
        f.update(z)

        inf.predict()
        inf.update(z)

        try:
            print(t, inf.P)
        except:
            pass

        # save data
        r.append(f.x[0, 0])
        r2.append(inf.x[0, 0])
        m.append(z)

    # assert np.allclose(f.x, inf.x), f'{t}: {f.x.T} {inf.x.T}'

    if DO_PLOT:
        plt.plot(m)
        plt.plot(r)
        plt.plot(r2)


def test_1d():
    global inf
    f = KalmanFilter(dim_x=2, dim_z=1)
    inf = InformationFilter(dim_x=2, dim_z=1)

    # ensure __repr__ doesn't assert
    str(inf)

    f.x = np.array([[2.0], [0.0]])  # initial state (location and velocity)

    inf.x = f.x.copy()
    f.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # state transition matrix

    inf.F = f.F.copy()
    f.H = np.array([[1.0, 0.0]])  # Measurement function
    inf.H = np.array([[1.0, 0.0]])  # Measurement function
    f.R *= 5  # state uncertainty
    inf.R_inv *= 1.0 / 5  # state uncertainty
    f.Q *= 0.0001  # process uncertainty
    inf.Q *= 0.0001

    m = []
    r = []
    r2 = []
    zs = []
    s = Saver(inf)
    for t in range(100):
        # create measurement = t plus white noise
        z = t + random.randn() * 20
        zs.append(z)

        # perform kalman filtering
        f.update(z)
        f.predict()

        inf.update(z)
        inf.predict()

        # save data
        r.append(f.x[0, 0])
        r2.append(inf.x[0, 0])
        m.append(z)
        print(inf.y)
        s.save()

        assert abs(f.x[0, 0] - inf.x[0, 0]) < 1.0e-12

    if DO_PLOT:
        plt.plot(m)
        plt.plot(r)
        plt.plot(r2)


def test_against_kf():
    inv = np.linalg.inv

    dt = 1.0
    IM = np.eye(2)
    Q = np.array([[0.25, 0.5], [0.5, 1]])

    F = np.array([[1, dt], [0, 1]])
    # QI = inv(Q)
    inv(IM)

    from bayesian_filters.kalman import InformationFilter

    # f = IF2(2, 1)
    r_std = 0.2
    R = np.array([[r_std * r_std]])
    RI = inv(R)

    """f.F = F.copy()
    f.H = np.array([[1, 0.]])
    f.RI = RI.copy()
    f.Q = Q.copy()
    f.IM = IM.copy()"""

    kf = KalmanFilter(2, 1)
    kf.F = F.copy()
    kf.H = np.array([[1, 0.0]])
    kf.R = R.copy()
    kf.Q = Q.copy()

    f0 = InformationFilter(2, 1)
    f0.F = F.copy()
    f0.H = np.array([[1, 0.0]])
    f0.R_inv = RI.copy()
    f0.Q = Q.copy()

    # f.IM = np.zeros((2,2))

    for i in range(1, 50):
        z = i + (np.random.rand() * r_std)
        f0.predict()
        # f.predict()
        kf.predict()

        f0.update(z)
        # f.update(z)
        kf.update(z)

        print(f0.x.T, kf.x.T)
        assert np.allclose(f0.x, kf.x)
        # assert np.allclose(f.x, kf.x)


def test_mahalanobis_property():
    """Test that mahalanobis property works correctly."""
    inf = InformationFilter(dim_x=2, dim_z=1)
    inf.x = np.array([[0.0], [0.0]])
    inf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    inf.H = np.array([[1.0, 0.0]])
    inf.R_inv = np.array([[1.0 / 5.0]])
    inf.Q = np.eye(2) * 0.0001
    inf.P_inv = np.eye(2) * 0.1

    # Do predict/update cycle
    inf.predict()
    inf.update(5.0)

    # Mahalanobis should be computed on first access
    maha = inf.mahalanobis
    assert isinstance(maha, float)
    assert maha > 0

    # Second access should return cached value
    maha2 = inf.mahalanobis
    assert maha == maha2


def test_likelihood_property():
    """Test that likelihood property works correctly."""
    inf = InformationFilter(dim_x=2, dim_z=1)
    inf.x = np.array([[0.0], [0.0]])
    inf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    inf.H = np.array([[1.0, 0.0]])
    inf.R_inv = np.array([[1.0 / 5.0]])
    inf.Q = np.eye(2) * 0.0001
    inf.P_inv = np.eye(2) * 0.1

    # Do predict/update cycle
    inf.predict()
    inf.update(5.0)

    # Likelihood should be computed on first access
    likelihood = inf.likelihood
    assert isinstance(likelihood, float)
    assert likelihood > 0
    assert likelihood <= 1.0

    # Second access should return cached value
    likelihood2 = inf.likelihood
    assert likelihood == likelihood2


def test_log_likelihood_property():
    """Test that log_likelihood property works correctly."""
    inf = InformationFilter(dim_x=2, dim_z=1)
    inf.x = np.array([[0.0], [0.0]])
    inf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    inf.H = np.array([[1.0, 0.0]])
    inf.R_inv = np.array([[1.0 / 5.0]])
    inf.Q = np.eye(2) * 0.0001
    inf.P_inv = np.eye(2) * 0.1

    # Do predict/update cycle
    inf.predict()
    inf.update(5.0)

    # Log-likelihood should be computed on first access
    log_likelihood = inf.log_likelihood
    assert isinstance(log_likelihood, float)
    assert log_likelihood <= 0  # Log of probability is negative

    # Second access should return cached value
    log_likelihood2 = inf.log_likelihood
    assert log_likelihood == log_likelihood2


def test_property_cache_invalidation_on_predict():
    """Test that property cache is invalidated on predict()."""
    inf = InformationFilter(dim_x=2, dim_z=1)
    inf.x = np.array([[0.0], [0.0]])
    inf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    inf.H = np.array([[1.0, 0.0]])
    inf.R_inv = np.array([[1.0 / 5.0]])
    inf.Q = np.eye(2) * 0.0001
    inf.P_inv = np.eye(2) * 0.1

    # Do first update
    inf.predict()
    inf.update(5.0)

    # Access properties to cache them
    maha1 = inf.mahalanobis
    likelihood1 = inf.likelihood
    log_likelihood1 = inf.log_likelihood

    # Do predict - this should invalidate cache
    inf.predict()
    inf.update(10.0)

    # Properties should be recomputed (different values)
    maha2 = inf.mahalanobis
    likelihood2 = inf.likelihood
    log_likelihood2 = inf.log_likelihood

    # Values should be different since we have a new measurement
    assert maha1 != maha2
    assert likelihood1 != likelihood2
    assert log_likelihood1 != log_likelihood2


def test_property_cache_invalidation_on_update():
    """Test that property cache is invalidated on update()."""
    inf = InformationFilter(dim_x=2, dim_z=1)
    inf.x = np.array([[0.0], [0.0]])
    inf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    inf.H = np.array([[1.0, 0.0]])
    inf.R_inv = np.array([[1.0 / 5.0]])
    inf.Q = np.eye(2) * 0.0001
    inf.P_inv = np.eye(2) * 0.1

    # Do first update
    inf.predict()
    inf.update(5.0)

    # Access properties to cache them
    maha1 = inf.mahalanobis

    # Do another update with different measurement
    inf.update(10.0)

    # Property should be recomputed (different value)
    maha2 = inf.mahalanobis

    # Value should be different since we have a different measurement
    assert maha1 != maha2


def test_likelihood_consistency():
    """Test that likelihood = exp(log_likelihood)."""
    inf = InformationFilter(dim_x=2, dim_z=1)
    inf.x = np.array([[0.0], [0.0]])
    inf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    inf.H = np.array([[1.0, 0.0]])
    inf.R_inv = np.array([[1.0 / 5.0]])
    inf.Q = np.eye(2) * 0.0001
    inf.P_inv = np.eye(2) * 0.1

    inf.predict()
    inf.update(5.0)

    likelihood = inf.likelihood
    log_likelihood = inf.log_likelihood

    # likelihood should be exp(log_likelihood)
    import math

    expected_likelihood = math.exp(log_likelihood)
    if expected_likelihood == 0:
        expected_likelihood = np.finfo(float).min

    assert np.isclose(likelihood, expected_likelihood)


def test_properties_with_none_measurement():
    """Test properties behavior when update is called with None."""
    inf = InformationFilter(dim_x=2, dim_z=1)
    inf.x = np.array([[0.0], [0.0]])
    inf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    inf.H = np.array([[1.0, 0.0]])
    inf.R_inv = np.array([[1.0 / 5.0]])
    inf.Q = np.eye(2) * 0.0001
    inf.P_inv = np.eye(2) * 0.1

    inf.predict()
    inf.update(None)

    # Properties should still be accessible (though may not be meaningful)
    # Just verify they don't throw exceptions and return None or valid values
    assert inf._mahalanobis is None
    assert inf._likelihood is None
    assert inf._log_likelihood is None


def test_repr_includes_properties():
    """Test that __repr__ includes the new properties."""
    inf = InformationFilter(dim_x=2, dim_z=1)
    inf.x = np.array([[0.0], [0.0]])
    inf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    inf.H = np.array([[1.0, 0.0]])
    inf.R_inv = np.array([[1.0 / 5.0]])
    inf.Q = np.eye(2) * 0.0001
    inf.P_inv = np.eye(2) * 0.1

    inf.predict()
    inf.update(5.0)

    repr_str = repr(inf)
    assert "mahalanobis" in repr_str
    assert "likelihood" in repr_str
    assert "log-likelihood" in repr_str


def test_property_values_match_kalman_filter():
    """Test that InformationFilter properties match KalmanFilter for same problem."""
    # Create equivalent filters
    kf = KalmanFilter(dim_x=2, dim_z=1)
    inf = InformationFilter(dim_x=2, dim_z=1)

    # Set up identical parameters
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    R = np.array([[5.0]])
    Q = np.eye(2) * 0.0001
    P = np.eye(2) * 10.0

    kf.x = np.array([[0.0], [0.0]])
    kf.F = F.copy()
    kf.H = H.copy()
    kf.R = R.copy()
    kf.Q = Q.copy()
    kf.P = P.copy()

    inf.x = np.array([[0.0], [0.0]])
    inf.F = F.copy()
    inf.H = H.copy()
    inf.R_inv = np.linalg.inv(R)
    inf.Q = Q.copy()
    inf.P_inv = np.linalg.inv(P)

    # Run one predict/update cycle
    z = 5.0
    kf.predict()
    kf.update(z)

    inf.predict()
    inf.update(z)

    # Properties should match
    assert np.isclose(kf.mahalanobis, inf.mahalanobis, rtol=1e-5)
    assert np.isclose(kf.likelihood, inf.likelihood, rtol=1e-5)
    assert np.isclose(kf.log_likelihood, inf.log_likelihood, rtol=1e-5)


if __name__ == "__main__":
    DO_PLOT = True
    # test_1d_0P()
    test_1d()
    test_against_kf()
    test_mahalanobis_property()
    test_likelihood_property()
    test_log_likelihood_property()
    test_property_cache_invalidation_on_predict()
    test_property_cache_invalidation_on_update()
    test_likelihood_consistency()
    test_properties_with_none_measurement()
    test_repr_includes_properties()
    test_property_values_match_kalman_filter()
