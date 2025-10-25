"""Comprehensive tests for KalmanFilter class.

This module provides extensive testing of the KalmanFilter implementation
across four categories:

1. Analytical Solution Tests: Verify against known analytical solutions
2. Property-Based Invariant Tests: Use Hypothesis to verify invariants
3. Edge Case and Error Handling: Test boundary conditions
4. Numerical Stability: Test long-running and ill-conditioned scenarios

These tests complement the existing test_kf.py by adding coverage for:
- Numerical accuracy against known solutions
- Property-based testing of matrix invariants
- Comprehensive edge case handling
- Long-term stability and convergence
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from bayesian_filters.kalman import KalmanFilter
from bayesian_filters.common import Q_discrete_white_noise
from bayesian_filters.testing_utils import (
    assert_matrix_psd,
    assert_matrix_symmetric,
    assert_filter_stable,
    ConstantVelocitySolution,
    ConstantAccelerationSolution,
)


# ============================================================================
# PART 1: ANALYTICAL SOLUTION TESTS
# ============================================================================


class TestKalmanFilterAnalyticalSolutions:
    """Test KalmanFilter against known analytical solutions."""

    @pytest.mark.numerical
    def test_constant_velocity_perfect_measurement(self):
        """Test tracking with perfect measurement (R=0, small Q).

        Setup: 1D constant velocity system with R=0 (perfect measurement), Q small.
        Expected: Position estimate should match measurements exactly.
        """
        dt = 0.1
        kf = KalmanFilter(dim_x=2, dim_z=1)

        # Setup constant velocity model
        kf.x = np.array([[0.0], [1.0]])
        kf.F = np.array([[1.0, dt], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 100.0
        kf.Q = np.eye(2) * 1e-8  # Very small process noise
        kf.R = np.array([[1e-10]])  # Very small measurement noise

        # Simulate constant velocity motion
        for t in range(10):
            # Analytical position
            z_true = ConstantVelocitySolution.position_at_time(t * dt, x0=0.0, v0=1.0)

            # Predict and update with near-perfect measurement
            kf.predict()
            kf.update(z_true)

            # Position should match measurements closely
            assert np.isclose(kf.x[0, 0], z_true, atol=0.01)

    @pytest.mark.numerical
    def test_constant_velocity_with_noise(self):
        """Test convergence with process and measurement noise.

        Setup: 1D constant velocity system with Q>0, R>0.
        Expected: Filter state should track within 2-3 sigma of truth.
        """
        dt = 0.1
        kf = KalmanFilter(dim_x=2, dim_z=1)

        # Setup constant velocity model with noise
        kf.x = np.array([[0.0], [1.0]])
        kf.F = np.array([[1.0, dt], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 100.0
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)
        kf.R = np.array([[1.0]])

        # Seed for reproducibility
        np.random.seed(42)

        # Simulate with noise
        errors = []
        for t in range(50):
            # Ground truth
            z_true = ConstantVelocitySolution.position_at_time(t * dt, x0=0.0, v0=1.0)

            # Noisy measurement
            z_measured = z_true + np.random.normal(0, 1.0)

            kf.predict()
            kf.update(z_measured)

            error = np.abs(kf.x[0, 0] - z_true)
            errors.append(error)

        # Final error should converge
        assert np.mean(errors[-10:]) < np.mean(errors[:10])

    @pytest.mark.numerical
    def test_constant_acceleration(self):
        """Test constant acceleration model.

        Setup: 1D constant acceleration system.
        Expected: Position estimates should be reasonable.
        """
        dt = 0.1
        kf = KalmanFilter(dim_x=3, dim_z=1)

        # Setup constant acceleration model
        kf.x = np.array([[0.0], [0.0], [1.0]])
        kf.F = np.array([[1.0, dt, 0.5 * dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0, 0.0]])
        kf.P = np.eye(3) * 100.0
        kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=0.01)
        kf.R = np.array([[0.1]])

        np.random.seed(42)

        for t in range(20):
            # Analytical solution
            z_true = ConstantAccelerationSolution.position_at_time(t * dt, x0=0.0, v0=0.0, a=1.0)

            z_measured = z_true + np.random.normal(0, np.sqrt(0.1))

            kf.predict()
            kf.update(z_measured)

            # Check estimates are reasonable (within measurement noise range)
            assert np.abs(kf.x[0, 0] - z_true) < 1.0
            assert np.isfinite(kf.x).all()

    @pytest.mark.numerical
    def test_2d_tracking(self):
        """Test 2D constant velocity tracking.

        Setup: 2D constant velocity system.
        Expected: Both x and y coordinates should be tracked reasonably.
        """
        dt = 0.1
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # Setup 2D constant velocity model
        kf.x = np.array([[0.0], [1.0], [0.0], [1.0]])
        F = np.eye(4)
        F[0, 1] = dt
        F[2, 3] = dt
        kf.F = F

        kf.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        kf.P = np.eye(4) * 100.0
        kf.Q = np.eye(4) * 0.01
        kf.R = np.eye(2)

        np.random.seed(42)

        for t in range(10):
            # Ground truth (moving diagonally)
            z_x = t * dt
            z_y = t * dt

            z_measured = np.array([[z_x], [z_y]]) + np.random.randn(2, 1)

            kf.predict()
            kf.update(z_measured)

            # Both coordinates should be tracked reasonably
            assert np.abs(kf.x[0, 0] - z_x) < 1.5
            assert np.abs(kf.x[2, 0] - z_y) < 1.5
            assert np.isfinite(kf.x).all()

    @pytest.mark.numerical
    def test_steady_state_convergence(self):
        """Test that filter covariance settles after initialization.

        Setup: 1D constant velocity system with constant noise.
        Expected: Covariance should settle (not growing indefinitely).
        """
        dt = 0.1
        kf = KalmanFilter(dim_x=2, dim_z=1)

        kf.x = np.array([[0.0], [1.0]])
        kf.F = np.array([[1.0, dt], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 100.0
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)
        kf.R = np.array([[1.0]])

        # Run filter to convergence
        covariances = []
        for _ in range(100):
            kf.predict()
            kf.update(np.array([[0.0]]))
            covariances.append(np.trace(kf.P))

        # Later traces should be stable (bounded)
        trace_early = np.mean(covariances[:10])
        trace_late = np.mean(covariances[-10:])

        # Both should be finite
        assert np.isfinite(trace_early)
        assert np.isfinite(trace_late)
        # Later trace should not be growing rapidly
        assert trace_late < trace_early * 10


# ============================================================================
# PART 2: PROPERTY-BASED INVARIANT TESTS
# ============================================================================


class TestKalmanFilterInvariants:
    """Test invariants that must always hold for KalmanFilter."""

    @pytest.mark.property
    def test_covariance_remains_symmetric(self, kf_1d_fixture):
        """Property: Covariance P must always remain symmetric."""
        kf = kf_1d_fixture

        for _ in range(20):
            kf.predict()
            assert_matrix_symmetric(kf.P, name="P after predict")

            kf.update(np.array([[0.0]]))
            assert_matrix_symmetric(kf.P, name="P after update")

    @pytest.mark.property
    def test_covariance_positive_semidefinite(self, kf_1d_fixture):
        """Property: Covariance P must always remain positive semi-definite."""
        kf = kf_1d_fixture

        for _ in range(20):
            kf.predict()
            assert_matrix_psd(kf.P, name="P after predict")

            kf.update(np.array([[np.random.randn()]]))
            assert_matrix_psd(kf.P, name="P after update")

    @pytest.mark.property
    @given(
        measurements=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_covariance_reduces_after_update(self, measurements, kf_1d_fixture):
        """Property: Trace of P should decrease after update (measurement reduces uncertainty)."""
        kf = kf_1d_fixture

        for z in measurements:
            kf.predict()
            trace_before_update = np.trace(kf.P)

            kf.update(np.array([[z]]))
            trace_after_update = np.trace(kf.P)

            # Update should not increase uncertainty (within numerical tolerance)
            assert trace_after_update <= trace_before_update + 1e-10

    @pytest.mark.property
    @given(
        measurements=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_state_remains_finite(self, measurements, kf_1d_fixture):
        """Property: State vector x must always contain finite values."""
        kf = kf_1d_fixture

        for z in measurements:
            kf.predict()
            assert np.isfinite(kf.x).all(), "State contains NaN or Inf after predict"

            kf.update(np.array([[z]]))
            assert np.isfinite(kf.x).all(), "State contains NaN or Inf after update"

    @pytest.mark.property
    def test_covariance_remains_finite(self, kf_1d_fixture):
        """Property: Covariance P must always contain finite values."""
        kf = kf_1d_fixture

        for _ in range(50):
            kf.predict()
            assert np.isfinite(kf.P).all(), "Covariance contains NaN or Inf after predict"

            kf.update(np.array([[np.random.randn()]]))
            assert np.isfinite(kf.P).all(), "Covariance contains NaN or Inf after update"

    @pytest.mark.property
    def test_state_shape_consistency(self, kf_2d_fixture):
        """Property: State shape must remain constant throughout filter lifecycle."""
        kf = kf_2d_fixture
        initial_shape = kf.x.shape

        for _ in range(20):
            kf.predict()
            assert kf.x.shape == initial_shape, "State shape changed after predict"

            kf.update(np.array([[0.0], [0.0]]))
            assert kf.x.shape == initial_shape, "State shape changed after update"

    @pytest.mark.property
    def test_covariance_shape_consistency(self, kf_2d_fixture):
        """Property: Covariance shape must remain constant."""
        kf = kf_2d_fixture
        initial_shape = kf.P.shape

        for _ in range(20):
            kf.predict()
            assert kf.P.shape == initial_shape, "Covariance shape changed"

            kf.update(np.array([[0.0], [0.0]]))
            assert kf.P.shape == initial_shape, "Covariance shape changed"

    @pytest.mark.property
    def test_kalman_gain_valid_dimensions(self, kf_1d_fixture):
        """Property: Kalman gain K must have correct dimensions."""
        kf = kf_1d_fixture

        for _ in range(10):
            kf.predict()
            kf.update(np.array([[0.0]]))

            # K should be dim_x × dim_z
            assert kf.K.shape == (kf.dim_x, kf.dim_z)

    @pytest.mark.property
    @given(
        scale_q=st.floats(min_value=0.001, max_value=10.0),
        scale_r=st.floats(min_value=0.001, max_value=10.0),
    )
    def test_uncertainty_growth_with_process_noise(self, scale_q, scale_r):
        """Property: Larger Q should result in larger steady-state covariance."""
        dt = 0.1

        def make_filter(q_scale, r_scale):
            kf = KalmanFilter(dim_x=2, dim_z=1)
            kf.x = np.array([[0.0], [1.0]])
            kf.F = np.array([[1.0, dt], [0.0, 1.0]])
            kf.H = np.array([[1.0, 0.0]])
            kf.P = np.eye(2) * 100.0
            kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1 * q_scale)
            kf.R = np.array([[1.0 * r_scale]])
            return kf

        # Run to steady state
        kf_low_q = make_filter(0.1, scale_r)
        kf_high_q = make_filter(1.0, scale_r)

        for _ in range(50):
            for kf in [kf_low_q, kf_high_q]:
                kf.predict()
                kf.update(np.array([[0.0]]))

        # Higher Q should lead to higher uncertainty
        trace_low_q = np.trace(kf_low_q.P)
        trace_high_q = np.trace(kf_high_q.P)
        assert trace_high_q > trace_low_q


# ============================================================================
# PART 3: EDGE CASE AND ERROR HANDLING TESTS
# ============================================================================


class TestKalmanFilterEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_zero_measurement_noise(self):
        """Test handling of zero measurement noise (R=0)."""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        kf.x = np.array([[0.0], [1.0]])
        kf.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 100.0
        kf.Q = np.zeros((2, 2))
        kf.R = np.zeros((1, 1))

        # Filter should not crash with R=0
        kf.predict()
        # Use a try-except to handle potential numerical issues
        try:
            kf.update(np.array([[1.0]]))
            # If update succeeds, the state should be updated
            assert np.isfinite(kf.x).all()
        except np.linalg.LinAlgError:
            # This is acceptable for zero measurement noise
            pass

    @pytest.mark.unit
    def test_zero_process_noise(self):
        """Test handling of zero process noise (Q=0)."""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        kf.x = np.array([[0.0], [1.0]])
        kf.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 100.0
        kf.Q = np.zeros((2, 2))
        kf.R = np.array([[1.0]])

        # Should handle Q=0 gracefully
        for _ in range(5):
            kf.predict()
            kf.update(np.array([[0.0]]))

            assert np.isfinite(kf.x).all()
            assert np.isfinite(kf.P).all()

    @pytest.mark.unit
    def test_very_small_measurement_noise(self):
        """Test handling of very small measurement noise."""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        kf.x = np.array([[0.0], [1.0]])
        kf.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 100.0
        kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.01)
        kf.R = np.array([[1e-10]])

        for _ in range(10):
            kf.predict()
            kf.update(np.array([[0.0]]))

            assert np.isfinite(kf.x).all()
            assert np.isfinite(kf.P).all()
            assert_matrix_psd(kf.P)

    @pytest.mark.unit
    def test_large_measurement_residual(self):
        """Test handling of outlier measurements."""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        kf.x = np.array([[0.0], [1.0]])
        kf.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 100.0
        kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.01)
        kf.R = np.array([[1.0]])

        # Update with a huge outlier
        kf.predict()
        kf.update(np.array([[1000.0]]))

        # Filter should still be valid
        assert np.isfinite(kf.x).all()
        assert np.isfinite(kf.P).all()
        assert_matrix_psd(kf.P)

    @pytest.mark.unit
    def test_repeated_predictions_without_update(self):
        """Test multiple predictions without measurements."""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        kf.x = np.array([[0.0], [1.0]])
        kf.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 10.0
        kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.01)
        kf.R = np.array([[1.0]])

        for _ in range(20):
            kf.predict()
            assert np.isfinite(kf.x).all()
            assert np.isfinite(kf.P).all()

    @pytest.mark.unit
    def test_single_dimension_filter(self):
        """Test degenerate case of 1x1 filter."""
        kf = KalmanFilter(dim_x=1, dim_z=1)

        kf.x = np.array([[0.0]])
        kf.F = np.array([[1.0]])
        kf.H = np.array([[1.0]])
        kf.P = np.array([[100.0]])
        kf.Q = np.array([[0.01]])
        kf.R = np.array([[1.0]])

        for _ in range(10):
            kf.predict()
            kf.update(np.array([[np.random.randn()]]))

            assert kf.x.shape == (1, 1)
            assert kf.P.shape == (1, 1)

    @pytest.mark.unit
    def test_high_dimensional_filter(self):
        """Test high-dimensional filter (10 states)."""
        dim = 10
        kf = KalmanFilter(dim_x=dim, dim_z=dim)

        kf.x = np.zeros((dim, 1))
        kf.F = np.eye(dim)
        kf.H = np.eye(dim)
        kf.P = np.eye(dim) * 100.0
        kf.Q = np.eye(dim) * 0.01
        kf.R = np.eye(dim)

        for _ in range(5):
            kf.predict()
            z = np.random.randn(dim, 1)
            kf.update(z)

            assert kf.x.shape == (dim, 1)
            assert kf.P.shape == (dim, dim)

    @pytest.mark.unit
    def test_mismatched_measurement_dimensions(self):
        """Test error handling for measurement dimension mismatch."""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        kf.x = np.array([[0.0], [1.0]])
        kf.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 100.0
        kf.Q = np.zeros((2, 2))
        kf.R = np.array([[1.0]])

        kf.predict()

        # Provide wrong measurement dimension
        with pytest.raises((ValueError, AssertionError)):
            kf.update(np.array([[1.0], [2.0]]))  # Should be 1D, not 2D

    @pytest.mark.unit
    def test_very_large_state_values(self):
        """Test handling of very large state values."""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        kf.x = np.array([[1e6], [1.0]])
        kf.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 100.0
        kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.01)
        kf.R = np.array([[1.0]])

        # Filter should handle large values
        for _ in range(10):
            kf.predict()
            kf.update(np.array([[1e6 + np.random.randn()]]))

            assert np.isfinite(kf.x).all()
            assert np.isfinite(kf.P).all()


# ============================================================================
# PART 4: NUMERICAL STABILITY TESTS
# ============================================================================


class TestKalmanFilterNumericalStability:
    """Test numerical stability over long runs and ill-conditioned systems."""

    @pytest.mark.slow
    @pytest.mark.numerical
    def test_long_running_stability_100_iterations(self):
        """Test stability over 100 iterations."""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        kf.x = np.array([[0.0], [1.0]])
        kf.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 100.0
        kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.1)
        kf.R = np.array([[1.0]])

        np.random.seed(42)

        for _ in range(100):
            kf.predict()
            z = np.random.randn() * np.sqrt(1.0) + np.sin(_)
            kf.update(np.array([[z]]))

            # Check numerical health at each iteration
            assert np.isfinite(kf.x).all(), f"NaN/Inf in x at iteration {_}"
            assert np.isfinite(kf.P).all(), f"NaN/Inf in P at iteration {_}"
            assert_matrix_psd(kf.P)

    @pytest.mark.slow
    @pytest.mark.numerical
    def test_long_running_stability_1000_iterations(self):
        """Test stability over 1000 iterations."""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        kf.x = np.array([[0.0], [1.0]])
        kf.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 10.0
        kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.05)
        kf.R = np.array([[1.0]])

        np.random.seed(42)

        traces = []
        for i in range(1000):
            kf.predict()
            z = 0.5 * np.sin(i * 0.01) + np.random.randn() * 0.1
            kf.update(np.array([[z]]))

            traces.append(np.trace(kf.P))

            if i % 100 == 0:
                # Periodically verify health
                assert np.isfinite(kf.P).all()
                assert_matrix_psd(kf.P)

        # Covariance should converge, not diverge
        late_mean = np.mean(traces[-100:])
        assert not np.isnan(late_mean)
        assert not np.isinf(late_mean)

    @pytest.mark.numerical
    def test_ill_conditioned_system(self):
        """Test filter stability with ill-conditioned system.

        Setup: State variables with very different magnitudes.
        Example: Position in meters (~1) and orientation in microradians (~1e-6).
        """
        kf = KalmanFilter(dim_x=2, dim_z=1)

        # State: [position_meters, orientation_microradians]
        kf.x = np.array([[100.0], [0.000001]])
        kf.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.diag([100.0, 1e-12])  # Very different scales
        kf.Q = np.diag([0.01, 1e-14])
        kf.R = np.array([[1.0]])

        for _ in range(20):
            kf.predict()
            kf.update(np.array([[100.0 + np.random.randn()]]))

            # Should remain stable despite ill conditioning
            assert np.isfinite(kf.x).all()
            assert np.isfinite(kf.P).all()
            assert_matrix_psd(kf.P)

    @pytest.mark.numerical
    def test_filter_with_changing_measurement_noise(self):
        """Test adaptive filter with changing measurement noise."""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        kf.x = np.array([[0.0], [1.0]])
        kf.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 100.0
        kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.01)

        np.random.seed(42)

        for i in range(50):
            kf.predict()

            # Measurement noise increases over time
            r = 1.0 + 0.01 * i
            kf.R = np.array([[r**2]])

            z = i * 0.1 + np.random.randn() * r
            kf.update(np.array([[z]]))

            assert np.isfinite(kf.x).all()
            assert np.isfinite(kf.P).all()

    @pytest.mark.numerical
    def test_filter_convergence_rate(self):
        """Test that filter converges at expected rate."""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        kf.x = np.array([[0.0], [1.0]])
        kf.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P = np.eye(2) * 1000.0
        kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.01)
        kf.R = np.array([[1.0]])

        traces = []
        for _ in range(50):
            kf.predict()
            kf.update(np.array([[0.0]]))
            traces.append(np.trace(kf.P))

        # Filter should converge monotonically (with some numerical tolerance)
        for i in range(1, len(traces)):
            assert traces[i] <= traces[i - 1] + 1e-6
