"""Testing utilities for bayesian_filters.

This package provides fixtures, helpers, and analytical solutions for testing
Kalman filters and related algorithms.

Submodules:
- fixtures: Common pytest fixtures for filter testing
- numerical: Analytical solutions and reference implementations
- helpers: Utility functions for test development
"""

from bayesian_filters.testing_utils.fixtures import (
    kf_1d,
    kf_2d,
    ekf_2d,
    ukf_3d,
    sensor_sim,
    noisy_measurements,
)
from bayesian_filters.testing_utils.numerical import (
    ConstantVelocitySolution,
    ConstantAccelerationSolution,
)
from bayesian_filters.testing_utils.helpers import (
    assert_matrix_psd,
    assert_matrix_symmetric,
    assert_filter_stable,
)

__all__ = [
    "kf_1d",
    "kf_2d",
    "ekf_2d",
    "ukf_3d",
    "sensor_sim",
    "noisy_measurements",
    "ConstantVelocitySolution",
    "ConstantAccelerationSolution",
    "assert_matrix_psd",
    "assert_matrix_symmetric",
    "assert_filter_stable",
]
