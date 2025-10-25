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
from scipy.linalg import sqrtm
from bayesian_filters.kalman.sigma_points import MerweScaledSigmaPoints, JulierSigmaPoints, SimplexSigmaPoints


class TestMerweScaledSigmaPoints:
    """Tests for MerweScaledSigmaPoints class."""

    def test_initialization(self):
        """Test basic initialization."""
        n = 3
        alpha = 0.1
        beta = 2.0
        kappa = 0.0

        points = MerweScaledSigmaPoints(n, alpha, beta, kappa)

        assert points.n == n
        assert points.alpha == alpha
        assert points.beta == beta
        assert points.kappa == kappa
        assert len(points.Wm) == 2 * n + 1
        assert len(points.Wc) == 2 * n + 1

    def test_num_sigmas(self):
        """Test that num_sigmas returns 2n+1."""
        for n in [1, 2, 5, 10]:
            points = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)
            assert points.num_sigmas() == 2 * n + 1

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1."""
        # Use alpha=1.0 for reasonable weights
        points = MerweScaledSigmaPoints(3, 1.0, 2.0, 0.0)
        # Wm should always sum to 1
        assert np.isclose(np.sum(points.Wm), 1.0)
        # Wc may not sum to exactly 1 due to beta parameter affecting the first weight

    def test_sigma_points_scalar_input(self):
        """Test sigma point generation with scalar input."""
        n = 1
        points = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        x = 5.0
        P = 1.0

        sigmas = points.sigma_points(x, P)

        assert sigmas.shape == (3, 1)  # 2*1+1 = 3 points
        assert sigmas[0, 0] == x  # First sigma point is the mean

    def test_sigma_points_array_input(self):
        """Test sigma point generation with array input."""
        n = 3
        points = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        x = np.array([1.0, 2.0, 3.0])
        P = np.eye(n)

        sigmas = points.sigma_points(x, P)

        assert sigmas.shape == (2 * n + 1, n)
        assert np.allclose(sigmas[0], x)  # First sigma point is the mean

    def test_sigma_points_scalar_covariance(self):
        """Test with scalar covariance (treated as P*I)."""
        n = 2
        points = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        x = np.array([0.0, 0.0])
        P_scalar = 2.0

        sigmas = points.sigma_points(x, P_scalar)

        assert sigmas.shape == (5, 2)
        # Verify first point is mean
        assert np.allclose(sigmas[0], x)

    def test_sigma_points_symmetry(self):
        """Test that sigma points are symmetric around mean."""
        n = 2
        points = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        x = np.array([5.0, 10.0])
        P = np.diag([1.0, 2.0])

        sigmas = points.sigma_points(x, P)

        # Check that points 1 to n and n+1 to 2n are symmetric
        for i in range(n):
            diff_pos = sigmas[i + 1] - x
            diff_neg = sigmas[n + i + 1] - x
            # They should be negatives of each other
            assert np.allclose(diff_pos, -diff_neg, atol=1e-10)

    def test_invalid_size_error(self):
        """Test error when x size doesn't match n."""
        points = MerweScaledSigmaPoints(3, 0.1, 2.0, 0.0)

        x = np.array([1.0, 2.0])  # Wrong size
        P = np.eye(3)

        with pytest.raises(ValueError, match="expected size"):
            points.sigma_points(x, P)

    def test_custom_sqrt_method(self):
        """Test with custom square root method."""
        n = 2
        points = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0, sqrt_method=sqrtm)

        x = np.array([0.0, 0.0])
        P = np.eye(n)

        sigmas = points.sigma_points(x, P)

        # Should still generate valid sigma points
        assert sigmas.shape == (2 * n + 1, n)

    def test_custom_subtract_function(self):
        """Test with custom subtraction function (for angle wrapping)."""

        def angle_subtract(a, b):
            """Subtract angles with wrapping."""
            diff = a - b
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            return diff

        n = 1
        points = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0, subtract=angle_subtract)

        x = np.array([3.0])  # Close to pi
        P = np.array([[0.1]])

        sigmas = points.sigma_points(x, P)

        # Should generate sigma points
        assert sigmas.shape == (3, 1)

    @pytest.mark.parametrize("alpha", [0.001, 0.1, 0.5, 1.0])
    def test_different_alpha_values(self, alpha):
        """Test with different alpha values (spread)."""
        n = 3
        points = MerweScaledSigmaPoints(n, alpha, 2.0, 0.0)

        x = np.zeros(n)
        P = np.eye(n)

        sigmas = points.sigma_points(x, P)

        # Larger alpha should spread points further from mean
        max_distance = np.max(np.linalg.norm(sigmas[1:] - x, axis=1))
        assert max_distance > 0

    def test_repr(self):
        """Test string representation."""
        points = MerweScaledSigmaPoints(2, 0.1, 2.0, 0.0)
        repr_str = repr(points)

        assert "MerweScaledSigmaPoints" in repr_str
        assert "alpha" in repr_str
        assert "beta" in repr_str


class TestJulierSigmaPoints:
    """Tests for JulierSigmaPoints class."""

    def test_initialization(self):
        """Test basic initialization."""
        n = 3
        kappa = 1.0

        points = JulierSigmaPoints(n, kappa)

        assert points.n == n
        assert points.kappa == kappa
        assert len(points.Wm) == 2 * n + 1
        assert len(points.Wc) == 2 * n + 1

    def test_num_sigmas(self):
        """Test that num_sigmas returns 2n+1."""
        for n in [1, 2, 5, 10]:
            points = JulierSigmaPoints(n)
            assert points.num_sigmas() == 2 * n + 1

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1."""
        points = JulierSigmaPoints(3, kappa=0.0)
        assert np.isclose(np.sum(points.Wm), 1.0)
        assert np.isclose(np.sum(points.Wc), 1.0)

    def test_weights_identical_for_mean_and_cov(self):
        """Test that Wm and Wc are identical for Julier."""
        points = JulierSigmaPoints(3, kappa=1.0)
        assert np.allclose(points.Wm, points.Wc)

    def test_sigma_points_generation(self):
        """Test basic sigma point generation."""
        n = 2
        points = JulierSigmaPoints(n, kappa=0.0)

        x = np.array([1.0, 2.0])
        P = np.eye(n)

        sigmas = points.sigma_points(x, P)

        assert sigmas.shape == (2 * n + 1, n)
        assert np.allclose(sigmas[0], x)

    def test_kappa_zero(self):
        """Test with kappa=0."""
        n = 2
        points = JulierSigmaPoints(n, kappa=0.0)

        x = np.zeros(n)
        P = np.eye(n)

        # Generate sigma points (validates input/output)
        _ = points.sigma_points(x, P)

        # First weight should be 0/(n+0) = 0
        assert np.isclose(points.Wm[0], 0.0)

    def test_kappa_positive(self):
        """Test with positive kappa."""
        n = 2
        kappa = 1.0
        points = JulierSigmaPoints(n, kappa=kappa)

        # First weight should be kappa/(n+kappa)
        expected_w0 = kappa / (n + kappa)
        assert np.isclose(points.Wm[0], expected_w0)

    @pytest.mark.parametrize("kappa", [-1.0, 0.0, 1.0, 3.0])
    def test_different_kappa_values(self, kappa):
        """Test with different kappa values."""
        n = 3
        points = JulierSigmaPoints(n, kappa=kappa)

        x = np.zeros(n)
        P = np.eye(n)

        sigmas = points.sigma_points(x, P)

        assert sigmas.shape == (2 * n + 1, n)
        assert np.sum(points.Wm) == pytest.approx(1.0)

    def test_compare_with_merwe(self):
        """Compare Julier with equivalent Merwe parameterization."""
        n = 3
        kappa = 0.0

        # Julier with kappa
        julier = JulierSigmaPoints(n, kappa=kappa)

        # Equivalent Merwe (alpha=1, beta=0, same kappa)
        merwe = MerweScaledSigmaPoints(n, alpha=1.0, beta=0.0, kappa=kappa)

        x = np.array([1.0, 2.0, 3.0])
        P = np.eye(n)

        sigmas_j = julier.sigma_points(x, P)
        sigmas_m = merwe.sigma_points(x, P)

        # Should produce similar sigma points
        assert np.allclose(sigmas_j, sigmas_m, atol=1e-10)


class TestSimplexSigmaPoints:
    """Tests for SimplexSigmaPoints class."""

    def test_initialization(self):
        """Test basic initialization."""
        n = 3
        alpha = 1.0

        points = SimplexSigmaPoints(n, alpha)

        assert points.n == n
        assert points.alpha == alpha
        assert len(points.Wm) == n + 1  # Only n+1 points!
        assert len(points.Wc) == n + 1

    def test_num_sigmas(self):
        """Test that num_sigmas returns n+1 (not 2n+1)."""
        for n in [1, 2, 5, 10]:
            points = SimplexSigmaPoints(n)
            assert points.num_sigmas() == n + 1  # Simplex uses fewer points

    def test_weights_are_uniform(self):
        """Test that all weights are equal (1/(n+1))."""
        n = 4
        points = SimplexSigmaPoints(n)

        expected_weight = 1.0 / (n + 1)
        assert np.allclose(points.Wm, expected_weight)
        assert np.allclose(points.Wc, expected_weight)

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1."""
        points = SimplexSigmaPoints(3)
        assert np.isclose(np.sum(points.Wm), 1.0)
        assert np.isclose(np.sum(points.Wc), 1.0)

    def test_sigma_points_generation(self):
        """Test basic sigma point generation."""
        n = 2
        points = SimplexSigmaPoints(n)

        x = np.array([1.0, 2.0])
        P = np.eye(n)

        sigmas = points.sigma_points(x, P)

        # Should return n+1 points
        assert sigmas.shape == (n + 1, n)

    def test_efficiency_fewer_points(self):
        """Test that simplex uses fewer points than Merwe/Julier."""
        n = 10

        simplex = SimplexSigmaPoints(n)
        merwe = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)

        # Simplex should have fewer points
        assert simplex.num_sigmas() < merwe.num_sigmas()
        assert simplex.num_sigmas() == n + 1
        assert merwe.num_sigmas() == 2 * n + 1

    def test_2d_case_forms_triangle(self):
        """Test that 2D simplex forms a triangle (3 points)."""
        n = 2
        points = SimplexSigmaPoints(n)

        x = np.zeros(n)
        P = np.eye(n)

        sigmas = points.sigma_points(x, P)

        # Should have exactly 3 points for 2D
        assert sigmas.shape == (3, 2)

    def test_different_alpha(self):
        """Test with different alpha values."""
        n = 3
        alpha = 2.0

        points = SimplexSigmaPoints(n, alpha=alpha)

        x = np.zeros(n)
        P = np.eye(n)

        sigmas = points.sigma_points(x, P)

        assert sigmas.shape == (n + 1, n)

    def test_repr(self):
        """Test string representation."""
        points = SimplexSigmaPoints(3, alpha=1.0)
        repr_str = repr(points)

        assert "SimplexSigmaPoints" in repr_str
        assert "alpha" in repr_str


class TestCrossComparison:
    """Cross-comparison tests between different sigma point methods."""

    def test_all_methods_produce_valid_points(self):
        """Test that all three methods produce valid sigma points."""
        n = 3
        x = np.array([1.0, 2.0, 3.0])
        P = np.diag([0.5, 1.0, 1.5])

        merwe = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)
        julier = JulierSigmaPoints(n, kappa=0.0)
        simplex = SimplexSigmaPoints(n)

        sigmas_m = merwe.sigma_points(x, P)
        sigmas_j = julier.sigma_points(x, P)
        sigmas_s = simplex.sigma_points(x, P)

        # All should have first dimension == n
        assert sigmas_m.shape[1] == n
        assert sigmas_j.shape[1] == n
        assert sigmas_s.shape[1] == n

        # Merwe and Julier have same number of points
        assert sigmas_m.shape[0] == sigmas_j.shape[0]
        # Simplex has fewer
        assert sigmas_s.shape[0] < sigmas_m.shape[0]

    def test_weights_normalized_all_methods(self):
        """Test that all methods have normalized weights."""
        n = 4

        merwe = MerweScaledSigmaPoints(n, 0.1, 2.0, 0.0)
        julier = JulierSigmaPoints(n, kappa=1.0)
        simplex = SimplexSigmaPoints(n)

        assert np.isclose(np.sum(merwe.Wm), 1.0)
        assert np.isclose(np.sum(julier.Wm), 1.0)
        assert np.isclose(np.sum(simplex.Wm), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
