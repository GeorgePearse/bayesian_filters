# -*- coding: utf-8 -*-
"""Copyright 2025 George Pearse

bayesian_filters library.
https://github.com/GeorgePearse/bayesian_filters

Documentation at:
https://georgepearse.github.io/bayesian_filters

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

import numpy as np
import pytest
from bayesian_filters.kalman.unscented_transform import unscented_transform
from bayesian_filters.kalman.sigma_points import MerweScaledSigmaPoints, JulierSigmaPoints, SimplexSigmaPoints


class TestUnscentedTransform:
    """Tests for unscented_transform function."""

    def test_basic_transform(self):
        """Test basic unscented transform without custom functions."""
        # Create sigma points
        n = 2
        points_gen = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        x = np.array([1.0, 2.0])
        P = np.eye(n) * 0.1

        sigmas = points_gen.sigma_points(x, P)

        # Transform
        x_mean, P_cov = unscented_transform(sigmas, points_gen.Wm, points_gen.Wc)

        # Mean should be close to original (identity transform)
        assert x_mean.shape == (n,)
        assert np.allclose(x_mean, x, atol=1e-10)

        # Covariance should be close to original
        assert P_cov.shape == (n, n)
        assert np.allclose(P_cov, P, atol=1e-5)

    def test_with_noise_cov(self):
        """Test transform with added noise covariance."""
        n = 2
        points_gen = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        x = np.array([0.0, 0.0])
        P = np.eye(n)

        sigmas = points_gen.sigma_points(x, P)

        # Add noise
        noise = np.eye(n) * 0.5

        x_mean, P_cov = unscented_transform(sigmas, points_gen.Wm, points_gen.Wc, noise_cov=noise)

        # Covariance should include noise
        expected_P = P + noise
        assert np.allclose(P_cov, expected_P, atol=1e-5)

    def test_custom_mean_function(self):
        """Test with custom mean function for angle states."""

        def angle_mean(sigmas, Wm):
            """Compute mean of angles using circular statistics."""
            sin_sum = np.sum(np.sin(sigmas[:, 0]) * Wm)
            cos_sum = np.sum(np.cos(sigmas[:, 0]) * Wm)
            angle = np.arctan2(sin_sum, cos_sum)
            return np.array([angle])

        n = 1
        points_gen = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        # Angle near pi
        x = np.array([3.0])
        P = np.array([[0.1]])

        sigmas = points_gen.sigma_points(x, P)

        x_mean, P_cov = unscented_transform(sigmas, points_gen.Wm, points_gen.Wc, mean_fn=angle_mean)

        # Should produce valid output
        assert x_mean.shape == (1,)
        assert P_cov.shape == (1, 1)

    def test_custom_residual_function(self):
        """Test with custom residual function for angle wrapping."""

        def angle_residual(a, b):
            """Compute angular difference with wrapping."""
            diff = a - b
            # Wrap to [-pi, pi]
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            return diff

        n = 1
        points_gen = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        x = np.array([3.0])
        P = np.array([[0.1]])

        sigmas = points_gen.sigma_points(x, P)

        x_mean, P_cov = unscented_transform(sigmas, points_gen.Wm, points_gen.Wc, residual_fn=angle_residual)

        assert x_mean.shape == (1,)
        assert P_cov.shape == (1, 1)

    def test_output_shapes(self):
        """Test that output shapes are correct for various dimensions."""
        for n in [1, 2, 5, 10]:
            points_gen = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

            x = np.zeros(n)
            P = np.eye(n)

            sigmas = points_gen.sigma_points(x, P)

            x_mean, P_cov = unscented_transform(sigmas, points_gen.Wm, points_gen.Wc)

            assert x_mean.shape == (n,)
            assert P_cov.shape == (n, n)

    def test_covariance_positive_definite(self):
        """Test that output covariance is positive definite."""
        n = 3
        points_gen = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        x = np.array([1.0, 2.0, 3.0])
        P = np.diag([1.0, 2.0, 3.0])

        sigmas = points_gen.sigma_points(x, P)

        x_mean, P_cov = unscented_transform(sigmas, points_gen.Wm, points_gen.Wc)

        # Check positive definiteness
        eigenvalues = np.linalg.eigvals(P_cov)
        assert np.all(eigenvalues > -1e-10)  # All non-negative (allowing small numerical errors)

    def test_covariance_symmetric(self):
        """Test that output covariance is symmetric."""
        n = 3
        points_gen = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        x = np.zeros(n)
        P = np.eye(n)

        sigmas = points_gen.sigma_points(x, P)

        x_mean, P_cov = unscented_transform(sigmas, points_gen.Wm, points_gen.Wc)

        assert np.allclose(P_cov, P_cov.T)

    def test_with_all_sigma_point_types(self):
        """Test that transform works with all sigma point generators."""
        n = 3
        x = np.array([1.0, 2.0, 3.0])
        P = np.eye(n)

        # Test with Merwe
        merwe = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)
        sigmas_m = merwe.sigma_points(x, P)
        x_m, P_m = unscented_transform(sigmas_m, merwe.Wm, merwe.Wc)
        assert x_m.shape == (n,)

        # Test with Julier
        julier = JulierSigmaPoints(n, kappa=0.0)
        sigmas_j = julier.sigma_points(x, P)
        x_j, P_j = unscented_transform(sigmas_j, julier.Wm, julier.Wc)
        assert x_j.shape == (n,)

        # Test with Simplex
        simplex = SimplexSigmaPoints(n)
        sigmas_s = simplex.sigma_points(x, P)
        x_s, P_s = unscented_transform(sigmas_s, simplex.Wm, simplex.Wc)
        assert x_s.shape == (n,)

    def test_large_dimension(self):
        """Test with large state dimension."""
        n = 20
        points_gen = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        x = np.random.randn(n)
        P = np.eye(n)

        sigmas = points_gen.sigma_points(x, P)

        x_mean, P_cov = unscented_transform(sigmas, points_gen.Wm, points_gen.Wc)

        assert x_mean.shape == (n,)
        assert P_cov.shape == (n, n)

    def test_identical_sigma_points(self):
        """Test with all sigma points identical (edge case)."""
        n = 2
        points_gen = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        # All sigma points are the same
        sigmas = np.array([[1.0, 2.0]] * (2 * n + 1))

        x_mean, P_cov = unscented_transform(sigmas, points_gen.Wm, points_gen.Wc)

        # Mean should be the common value
        assert np.allclose(x_mean, np.array([1.0, 2.0]))

        # Covariance should be near zero
        assert np.allclose(P_cov, np.zeros((n, n)), atol=1e-10)

    def test_residual_fn_none_uses_subtract(self):
        """Test that residual_fn=None uses np.subtract."""
        n = 2
        points_gen = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        x = np.zeros(n)
        P = np.eye(n)

        sigmas = points_gen.sigma_points(x, P)

        # These should give same result
        x1, P1 = unscented_transform(sigmas, points_gen.Wm, points_gen.Wc, residual_fn=None)
        x2, P2 = unscented_transform(sigmas, points_gen.Wm, points_gen.Wc, residual_fn=np.subtract)

        assert np.allclose(x1, x2)
        assert np.allclose(P1, P2)

    def test_numerical_stability(self):
        """Test numerical stability with ill-conditioned covariance."""
        n = 3
        points_gen = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        x = np.zeros(n)
        # Ill-conditioned covariance
        P = np.diag([1e-6, 1.0, 1e6])

        sigmas = points_gen.sigma_points(x, P)

        x_mean, P_cov = unscented_transform(sigmas, points_gen.Wm, points_gen.Wc)

        # Should still produce valid output
        assert np.all(np.isfinite(x_mean))
        assert np.all(np.isfinite(P_cov))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
