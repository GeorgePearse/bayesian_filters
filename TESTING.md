# Testing Guide for Bayesian Filters

This document provides guidance on running tests, using testing utilities, and interpreting test results.

## Quick Start

### Running All Tests

```bash
pytest
```

### Running Tests with Coverage

```bash
pytest --cov=bayesian_filters --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Running Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Property-based tests (Hypothesis)
pytest -m property

# Numerical accuracy tests
pytest -m numerical

# Benchmarks
pytest -m benchmark --benchmark-only

# Slow tests (>1 second)
pytest -m slow

# Everything except slow tests
pytest -m "not slow"
```

## Test Organization

Tests are organized by module and test type:

```
bayesian_filters/
├── kalman/
│   ├── test_*.py              # Unit tests
│   ├── test_*_property.py     # Property-based tests
│   └── test_*_benchmark.py    # Performance benchmarks
├── extended_kalman/
├── unscented_kalman/
├── common/
└── stats/
```

## Using Testing Utilities

### Fixtures

Pre-configured filter fixtures are available via pytest:

```python
def test_with_fixture(kf_1d_fixture):
    """Test using a 1D Kalman filter fixture."""
    kf = kf_1d_fixture
    # kf has state [position, velocity], dt=0.1
    assert kf.x.shape == (2, 1)

def test_2d_filter(kf_2d_fixture):
    """Test using a 2D Kalman filter fixture."""
    kf = kf_2d_fixture
    # kf has state [x, vx, y, vy]
    assert kf.x.shape == (4, 1)
```

**Available fixtures:**
- `kf_1d_fixture`: 1D constant-velocity Kalman filter
- `kf_2d_fixture`: 2D constant-velocity Kalman filter
- `ekf_2d_fixture`: 2D Extended Kalman filter with range measurement
- `ukf_3d_fixture`: 3D Unscented Kalman filter
- `sensor_sim_fixture`: Simple 1D sensor simulator

### Helper Functions

Test assertion utilities for validating filter properties:

```python
from bayesian_filters.testing_utils import (
    assert_matrix_psd,
    assert_matrix_symmetric,
    assert_filter_stable,
)

def test_covariance_properties(kf_1d_fixture):
    """Verify filter covariance matrix properties."""
    kf = kf_1d_fixture

    # Check covariance is positive semi-definite
    assert_matrix_psd(kf.P, name="Initial covariance")

    # Check covariance is symmetric
    assert_matrix_symmetric(kf.P, name="Initial covariance")

    # Check filter stability
    kf.predict()
    assert_filter_stable(kf.P, max_variance=1e10)
```

### Numerical Solutions

Analytical solutions for validation:

```python
from bayesian_filters.testing_utils import ConstantVelocitySolution

def test_against_analytical_solution():
    """Compare filter estimate against analytical solution."""
    dt = 0.1
    x0, v0 = 0.0, 1.0
    t = 1.0

    # Analytical position at t=1.0
    analytical_pos = ConstantVelocitySolution.position_at_time(t, x0, v0)

    # Compare with filter estimate
    assert abs(filter_estimate - analytical_pos) < 0.1
```

## Test Types

### Unit Tests

Fast, isolated tests for individual functions or methods.

```python
@pytest.mark.unit
def test_kalman_gain_computation(kf_1d_fixture):
    """Test Kalman gain computation."""
    kf = kf_1d_fixture
    # Test implementation
    assert K.shape == (2, 1)
```

### Property-Based Tests

Use Hypothesis to generate random test cases and verify invariants:

```python
from hypothesis import given, strategies as st

@pytest.mark.property
@given(
    measurement=st.floats(min_value=-100, max_value=100),
    noise_cov=st.floats(min_value=0.1, max_value=10.0),
)
def test_filter_convergence(measurement, noise_cov):
    """Test that filter estimates converge with Hypothesis."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    # ... setup ...

    # Filter should produce reasonable estimates
    kf.update(measurement)
    assert np.isfinite(kf.x).all()
    assert kf.P[0, 0] > 0
```

### Numerical Accuracy Tests

Verify filter accuracy against known solutions:

```python
@pytest.mark.numerical
def test_constant_velocity_tracking():
    """Test filter tracks constant velocity motion accurately."""
    # Known motion
    ground_truth = ConstantVelocitySolution.position_at_time(t=1.0)

    # Filter estimate
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.0], [1.0]])
    kf.predict()

    # Error should be small
    error = abs(kf.x[0, 0] - ground_truth)
    assert error < 0.01
```

### Performance Benchmarks

Measure filter performance and detect regressions:

```python
@pytest.mark.benchmark
def test_kalman_filter_predict_benchmark(benchmark, kf_1d_fixture):
    """Benchmark Kalman filter prediction speed."""
    kf = kf_1d_fixture

    # Measure predict() speed
    result = benchmark(kf.predict)

    # Should complete quickly
    assert benchmark.stats.mean < 0.001  # < 1 ms
```

## Coverage Requirements

Test coverage targets by module:

| Module | Target | Notes |
|--------|--------|-------|
| kalman.KalmanFilter | 95% | Core algorithm |
| kalman.ExtendedKalmanFilter | 90% | Includes nonlinearity |
| kalman.UnscentedKalmanFilter | 90% | Sigma point algorithm |
| common | 85% | Utility functions |
| stats | 80% | Statistical functions |
| Other modules | 75% | Supporting code |

## Running Coverage Analysis

### Generate Coverage Report

```bash
pytest --cov=bayesian_filters --cov-report=html
open htmlcov/index.html
```

### Show Missing Lines

```bash
pytest --cov=bayesian_filters --cov-report=term-missing
```

### Coverage for Specific Module

```bash
pytest --cov=bayesian_filters.kalman --cov-report=term-missing
```

## Parallel Test Execution

Run tests in parallel for faster feedback:

```bash
# Run on all CPU cores
pytest -n auto

# Run on specific number of cores
pytest -n 4
```

## Continuous Integration

Tests are automatically run on GitHub Actions for:
- All push events to testing branches
- All pull requests
- Multiple Python versions (3.9, 3.10, 3.11, 3.12)
- Coverage collection and reporting
- Performance benchmarks on master branch

See `.github/workflows/test-coverage.yml` for full CI/CD configuration.

## Debugging Test Failures

### Verbose Output

```bash
pytest -vv --tb=long test_file.py::test_function
```

### Drop into Debugger

```bash
pytest --pdb test_file.py::test_function
```

### Show Print Statements

```bash
pytest -s test_file.py::test_function
```

### Run with Random Seed

```bash
# Reproducible randomization
pytest --seed=12345

# Get seed from last failure
pytest --lastfailed --seed=12345
```

## Writing New Tests

### Test Structure

```python
"""Tests for kalman.py module.

This module tests the KalmanFilter class including:
- Filter initialization
- Prediction step
- Update step
- Covariance matrix properties
"""

import pytest
import numpy as np
from bayesian_filters.kalman import KalmanFilter
from bayesian_filters.testing_utils import assert_matrix_psd

@pytest.mark.unit
class TestKalmanFilterInitialization:
    """Test KalmanFilter initialization."""

    def test_state_initialization(self, kf_1d_fixture):
        """Verify initial state is set correctly."""
        kf = kf_1d_fixture
        assert kf.x.shape == (2, 1)
        assert kf.x[0, 0] == 0.0
        assert kf.x[1, 0] == 1.0

    def test_covariance_initialization(self, kf_1d_fixture):
        """Verify initial covariance is positive definite."""
        kf = kf_1d_fixture
        assert_matrix_psd(kf.P)

@pytest.mark.unit
def test_kalman_filter_predict(kf_1d_fixture):
    """Test prediction step maintains covariance properties."""
    kf = kf_1d_fixture

    # Predict
    kf.predict()

    # Covariance should still be valid
    assert_matrix_psd(kf.P)
    assert np.isfinite(kf.x).all()
    assert np.isfinite(kf.P).all()
```

### Guidelines

1. **Use fixtures** for common setup (kf_1d_fixture, sensor_sim_fixture, etc.)
2. **Test invariants** - verify properties that should always hold
3. **Use markers** - @pytest.mark.unit, @pytest.mark.numerical, etc.
4. **Clear names** - test function names should describe what they test
5. **Assertions** - use helper functions from testing_utils when possible
6. **Avoid randomness** - use fixed seeds or Hypothesis for controlled randomness

## Interpreting Coverage Reports

The HTML coverage report (`htmlcov/index.html`) shows:

- **Green lines**: Covered by tests
- **Red lines**: Not covered by tests
- **Yellow lines**: Partially covered
- **Missing branches**: Shows which if/else branches aren't tested

Aim for:
- **High line coverage** (80-95%)
- **Branch coverage** for critical paths (>90%)
- **All public APIs** tested

## Common Issues

### "No tests ran"
```bash
# Check test discovery
pytest --collect-only

# Verify test file naming (test_*.py or *_test.py)
# Verify test function naming (test_* prefix)
```

### "Fixtures not found"
```bash
# Check conftest.py is in project root
# Verify fixture is exported in __init__.py
# Use pytest --fixtures to list available fixtures
pytest --fixtures | grep kalman
```

### "Coverage missing"
```bash
# Reinstall in development mode
pip install -e .[dev]

# Check pyproject.toml has omit patterns correct
# Run with --no-cov-on-fail
pytest --cov --no-cov-on-fail
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
- Testing Plan: See `TESTING_PLAN.md` for comprehensive testing strategy
