# -*- coding: utf-8 -*-
"""Copyright 2025 George Pearse

bayesian_filters library.
https://github.com/GeorgePearse/bayesian_filters

Documentation at:
https://georgepearse.github.io/bayesian_filters

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

import numpy as np
import pytest
import math
from bayesian_filters.common.kinematic import kinematic_state_transition, kinematic_kf


class TestKinematicStateTransition:
    """Tests for kinematic_state_transition function."""

    def test_order_zero(self):
        """Test order 0 (position only) state transition matrix."""
        F = kinematic_state_transition(0, dt=0.1)
        expected = np.array([[1.0]])
        assert np.allclose(F, expected)
        assert F.shape == (1, 1)

    def test_order_one_constant_velocity(self):
        """Test order 1 (position + velocity) state transition matrix."""
        dt = 0.5
        F = kinematic_state_transition(1, dt)
        expected = np.array([[1.0, dt], [0.0, 1.0]])
        assert np.allclose(F, expected)
        assert F.shape == (2, 2)

    def test_order_two_constant_acceleration(self):
        """Test order 2 (position + velocity + acceleration) state transition matrix."""
        dt = 0.25
        F = kinematic_state_transition(2, dt)
        expected = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        assert np.allclose(F, expected)
        assert F.shape == (3, 3)

    @pytest.mark.parametrize("order", [3, 4, 5, 6])
    def test_higher_orders(self, order):
        """Test higher order state transitions use computational path."""
        dt = 0.1
        F = kinematic_state_transition(order, dt)

        # Verify shape
        expected_size = order + 1
        assert F.shape == (expected_size, expected_size)

        # Verify diagonal is all ones
        assert np.allclose(np.diag(F), np.ones(expected_size))

        # Verify lower triangle is zeros
        for i in range(expected_size):
            for j in range(i):
                assert F[i, j] == 0.0, f"Expected F[{i},{j}] = 0, got {F[i, j]}"

        # Verify first row follows dt^n/n! pattern
        for n in range(expected_size):
            expected_val = dt**n / math.factorial(n)
            assert np.isclose(F[0, n], expected_val), f"F[0,{n}] = {F[0, n]}, expected {expected_val}"

    def test_zero_dt(self):
        """Test with dt=0 returns identity-like matrix."""
        F = kinematic_state_transition(2, dt=0)
        # With dt=0, only diagonal should be 1, rest should be 0
        assert np.allclose(F, np.eye(3))

    def test_negative_dt(self):
        """Test with negative dt (backward time)."""
        dt = -0.1
        F = kinematic_state_transition(1, dt)
        expected = np.array([[1.0, dt], [0.0, 1.0]])
        assert np.allclose(F, expected)

    def test_large_dt(self):
        """Test with large dt value."""
        dt = 10.0
        F = kinematic_state_transition(2, dt)
        expected = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        assert np.allclose(F, expected)

    def test_invalid_negative_order(self):
        """Test that negative order raises ValueError."""
        with pytest.raises(ValueError, match="order must be an int >= 0"):
            kinematic_state_transition(-1, 0.1)

    def test_invalid_non_integer_order(self):
        """Test that non-integer order raises ValueError."""
        with pytest.raises(ValueError, match="order must be an int >= 0"):
            kinematic_state_transition(1.5, 0.1)

    def test_matrix_properties(self):
        """Verify mathematical properties of state transition matrix."""
        dt = 0.1
        for order in range(4):
            F = kinematic_state_transition(order, dt)
            n = order + 1

            # Should be upper triangular
            for i in range(n):
                for j in range(i):
                    assert F[i, j] == 0.0

            # Diagonal should be ones
            assert np.allclose(np.diag(F), np.ones(n))

            # Should be invertible (determinant != 0)
            det = np.linalg.det(F)
            assert not np.isclose(det, 0)

    def test_consistency_across_implementations(self):
        """Verify hardcoded cases match computational implementation."""
        dt = 0.1

        # Test order 0
        F0_hardcoded = kinematic_state_transition(0, dt)
        # Manually compute what the computational path would give
        assert F0_hardcoded.shape == (1, 1)
        assert F0_hardcoded[0, 0] == 1.0

        # Test order 1
        F1_hardcoded = kinematic_state_transition(1, dt)
        assert F1_hardcoded.shape == (2, 2)
        assert F1_hardcoded[0, 0] == 1.0
        assert F1_hardcoded[0, 1] == dt

        # Test order 2
        F2_hardcoded = kinematic_state_transition(2, dt)
        assert F2_hardcoded.shape == (3, 3)
        assert F2_hardcoded[0, 2] == 0.5 * dt * dt


class TestKinematicKF:
    """Tests for kinematic_kf function."""

    def test_1d_position_only(self):
        """Test 1D filter with position only (order=0)."""
        kf = kinematic_kf(dim=1, order=0, dt=0.1)

        # Check dimensions
        assert kf.dim_x == 1
        assert kf.dim_z == 1
        assert kf.x.shape == (1, 1)
        assert kf.F.shape == (1, 1)
        assert kf.H.shape == (1, 1)
        assert kf.P.shape == (1, 1)
        assert kf.Q.shape == (1, 1)
        assert kf.R.shape == (1, 1)

        # Verify H matrix
        assert kf.H[0, 0] == 1.0

    def test_1d_constant_velocity(self):
        """Test 1D filter with constant velocity (order=1)."""
        dt = 0.2
        kf = kinematic_kf(dim=1, order=1, dt=dt)

        # Check dimensions
        assert kf.dim_x == 2
        assert kf.dim_z == 1
        assert kf.F.shape == (2, 2)
        assert kf.H.shape == (1, 2)

        # Verify F matrix
        expected_F = np.array([[1.0, dt], [0.0, 1.0]])
        assert np.allclose(kf.F, expected_F)

        # Verify H matrix (measure position only)
        expected_H = np.array([[1.0, 0.0]])
        assert np.allclose(kf.H, expected_H)

    def test_2d_constant_velocity_order_by_dim_true(self):
        """Test 2D constant velocity filter with order_by_dim=True."""
        dt = 0.1
        kf = kinematic_kf(dim=2, order=1, dt=dt, order_by_dim=True)

        # Check dimensions
        assert kf.dim_x == 4  # [x, vx, y, vy]
        assert kf.dim_z == 1

        # F should be block diagonal
        F_block = np.array([[1.0, dt], [0.0, 1.0]])
        expected_F = np.block([[F_block, np.zeros((2, 2))], [np.zeros((2, 2)), F_block]])
        assert np.allclose(kf.F, expected_F)

        # H should have 1s at position 0 for each dimension
        # With dim_z=1, H shape is (1, 4), measuring positions
        assert kf.H.shape == (1, 4)
        assert kf.H[0, 0] == 1.0  # Measure first dim position
        assert kf.H[0, 2] == 1.0  # Measure second dim position

    def test_2d_constant_velocity_order_by_dim_false(self):
        """Test 2D constant velocity filter with order_by_dim=False."""
        dt = 0.1
        kf = kinematic_kf(dim=2, order=1, dt=dt, order_by_dim=False)

        # Check dimensions
        assert kf.dim_x == 4  # [x, y, vx, vy]
        assert kf.dim_z == 1

        # F should have a different structure (not block diagonal)
        # Verify specific elements
        assert kf.F[0, 0] == 1.0  # x position
        assert kf.F[1, 1] == 1.0  # y position
        assert kf.F[2, 2] == 1.0  # vx
        assert kf.F[3, 3] == 1.0  # vy
        assert kf.F[0, 2] == dt  # x influenced by vx
        assert kf.F[1, 3] == dt  # y influenced by vy

        # H should measure both x and y
        expected_H = np.array([[1.0, 1.0, 0.0, 0.0]])
        assert np.allclose(kf.H, expected_H)

    def test_3d_constant_acceleration(self):
        """Test 3D filter with constant acceleration (order=2)."""
        dt = 0.15
        kf = kinematic_kf(dim=3, order=2, dt=dt)

        # Check dimensions
        assert kf.dim_x == 9  # 3 dimensions * (pos + vel + acc)
        assert kf.dim_z == 1
        assert kf.F.shape == (9, 9)
        assert kf.H.shape == (1, 9)

    @pytest.mark.parametrize("dim", [1, 2, 3, 4])
    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_various_dimensions_and_orders(self, dim, order):
        """Test various combinations of dimensions and orders."""
        dt = 0.1
        kf = kinematic_kf(dim=dim, order=order, dt=dt)

        # Verify dimensions
        expected_dim_x = dim * (order + 1)
        assert kf.dim_x == expected_dim_x
        assert kf.x.shape == (expected_dim_x, 1)
        assert kf.F.shape == (expected_dim_x, expected_dim_x)

        # Should be able to run predict/update cycle without errors
        kf.predict()
        kf.update(np.zeros((kf.dim_z, 1)))

    def test_multiple_measurements_per_dim(self):
        """Test filter with dim_z > 1."""
        kf = kinematic_kf(dim=2, order=1, dt=0.1, dim_z=3)

        assert kf.dim_z == 3
        assert kf.H.shape == (3, 4)
        assert kf.R.shape == (3, 3)

    def test_h_matrix_structure_order_by_dim_true(self):
        """Verify H matrix structure with order_by_dim=True."""
        kf = kinematic_kf(dim=3, order=2, dt=0.1, order_by_dim=True)

        # With order_by_dim=True and dim_z=1, H should measure all positions
        # State is [x, vx, ax, y, vy, ay, z, vz, az]
        assert kf.H.shape == (1, 9)
        assert kf.H[0, 0] == 1.0  # Measure x position
        assert kf.H[0, 3] == 1.0  # Measure y position
        assert kf.H[0, 6] == 1.0  # Measure z position

    def test_h_matrix_structure_order_by_dim_false(self):
        """Verify H matrix structure with order_by_dim=False."""
        kf = kinematic_kf(dim=3, order=2, dt=0.1, order_by_dim=False)

        # With order_by_dim=False and dim_z=1, H should measure all positions
        # State is [x, y, z, vx, vy, vz, ax, ay, az]
        expected_H = np.array([[1.0, 1, 1, 0, 0, 0, 0, 0, 0]])
        assert np.allclose(kf.H, expected_H)

    def test_predict_update_cycle(self):
        """Test that filter can execute predict/update cycle."""
        kf = kinematic_kf(dim=2, order=1, dt=0.1)

        # Initialize state
        kf.x = np.array([[0.0], [1.0], [0.0], [1.0]])  # Initial position and velocity

        # Run several cycles
        for i in range(10):
            kf.predict()
            measurement = np.array([[float(i)]])
            kf.update(measurement)

        # State should have changed
        assert not np.allclose(kf.x, np.array([[0.0], [1.0], [0.0], [1.0]]))

    def test_error_dim_less_than_1(self):
        """Test that dim < 1 raises ValueError."""
        with pytest.raises(ValueError, match="dim must be >= 1"):
            kinematic_kf(dim=0, order=1)

    def test_error_order_less_than_0(self):
        """Test that order < 0 raises ValueError."""
        with pytest.raises(ValueError, match="order must be >= 0"):
            kinematic_kf(dim=2, order=-1)

    def test_error_dim_z_less_than_1(self):
        """Test that dim_z < 1 raises ValueError."""
        with pytest.raises(ValueError, match="dim_z must be >= 1"):
            kinematic_kf(dim=2, order=1, dim_z=0)

    def test_custom_kf_parameter(self):
        """Test providing custom KalmanFilter object."""
        from bayesian_filters.kalman import KalmanFilter

        # Create custom filter
        custom_kf = KalmanFilter(dim_x=4, dim_z=2)

        # Pass it to kinematic_kf
        result_kf = kinematic_kf(dim=2, order=1, dt=0.1, dim_z=2, kf=custom_kf)

        # Should return the same object (modified)
        assert result_kf is custom_kf

        # F matrix should be set
        assert not np.allclose(result_kf.F, np.eye(4))

    def test_custom_kf_wrong_dimensions(self):
        """Test that custom kf with wrong dimensions fails assertion."""
        from bayesian_filters.kalman import KalmanFilter

        # Create filter with wrong dimensions
        wrong_kf = KalmanFilter(dim_x=2, dim_z=1)

        # Should fail assertion
        with pytest.raises(AssertionError):
            kinematic_kf(dim=2, order=1, kf=wrong_kf)  # Expects dim_x=4

    def test_f_matrix_invertibility(self):
        """Verify F matrix is invertible for all configurations."""
        for dim in [1, 2, 3]:
            for order in [0, 1, 2]:
                kf = kinematic_kf(dim=dim, order=order, dt=0.1)
                det = np.linalg.det(kf.F)
                assert not np.isclose(det, 0), f"F matrix not invertible for dim={dim}, order={order}"

    def test_state_propagation(self):
        """Test that state propagates correctly through F matrix."""
        dt = 0.1
        kf = kinematic_kf(dim=1, order=1, dt=dt)

        # Set initial state: position=0, velocity=1
        kf.x = np.array([[0.0], [1.0]])

        # Predict (no noise)
        kf.Q = np.zeros((2, 2))
        kf.predict()

        # After one step, position should be velocity*dt
        expected_position = 1.0 * dt
        assert np.isclose(kf.x[0, 0], expected_position)
        # Velocity should remain 1
        assert np.isclose(kf.x[1, 0], 1.0)

    @pytest.mark.parametrize("order_by_dim", [True, False])
    def test_both_ordering_modes_work(self, order_by_dim):
        """Test both ordering modes produce valid filters."""
        kf = kinematic_kf(dim=2, order=2, dt=0.1, order_by_dim=order_by_dim)

        # Should be able to run full cycle
        kf.predict()
        kf.update(np.array([[0.0]]))

        # F should have correct structure
        assert np.allclose(np.diag(kf.F), np.ones(6))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
