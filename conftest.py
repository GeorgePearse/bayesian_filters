"""Root pytest configuration for bayesian_filters."""

import pytest
import numpy as np
from bayesian_filters.testing_utils.fixtures import (
    kf_1d,
    kf_2d,
    ekf_2d,
    ukf_3d,
    sensor_sim,
    noisy_measurements,
)


# Export fixtures so they're available to all tests
pytest_plugins = ["pytest_benchmark"]


@pytest.fixture
def kf_1d_fixture():
    """1D Kalman filter fixture."""
    return kf_1d()


@pytest.fixture
def kf_2d_fixture():
    """2D Kalman filter fixture."""
    return kf_2d()


@pytest.fixture
def ekf_2d_fixture():
    """2D Extended Kalman filter fixture."""
    return ekf_2d()


@pytest.fixture
def ukf_3d_fixture():
    """3D Unscented Kalman filter fixture."""
    return ukf_3d()


@pytest.fixture
def sensor_sim_fixture():
    """Sensor simulator fixture."""
    return sensor_sim()


def pytest_configure(config):
    """Configure pytest."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow (>1 second)")
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark")
    config.addinivalue_line("markers", "property: mark test as property-based (hypothesis)")
    config.addinivalue_line("markers", "numerical: mark test as numerical accuracy test")


def pytest_collection_modifyitems(config, items):
    """Modify collected items to add markers."""
    for item in items:
        # Auto-mark slow tests
        if "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.benchmark)

        # Auto-mark property tests
        if "property" in item.nodeid:
            item.add_marker(pytest.mark.property)


# Set random seed for reproducibility
np.random.seed(42)
