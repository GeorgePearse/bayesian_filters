# -*- coding: utf-8 -*-
"""Copyright 2025 George Pearse

bayesian_filters library.
https://github.com/GeorgePearse/bayesian_filters

Documentation at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

import numpy as np
import pytest
from bayesian_filters.monte_carlo.resampling import (
    residual_resample,
    stratified_resample,
    systematic_resample,
    multinomial_resample,
)


class TestResidualResample:
    """Tests for residual_resample function."""

    def test_uniform_weights(self):
        """Test with uniform weights."""
        N = 100
        weights = np.ones(N) / N

        indexes = residual_resample(weights)

        assert len(indexes) == N
        assert np.all(indexes >= 0)
        assert np.all(indexes < N)

    def test_skewed_weights_one_dominant(self):
        """Test with one weight much larger than others."""
        N = 100
        weights = np.ones(N) * 0.01
        weights[0] = 0.99  # One dominant particle
        weights /= np.sum(weights)  # Normalize

        indexes = residual_resample(weights)

        # Most indexes should point to particle 0
        assert np.sum(indexes == 0) > N / 2

    def test_preserves_weight_distribution_statistically(self):
        """Test that resampling preserves weight distribution over many runs."""
        np.random.seed(42)
        N = 10  # Smaller N for more reliable statistics
        # Use more skewed weights for clearer test
        weights = np.array([0.4, 0.3, 0.2, 0.1] + [0.0] * 6)

        # Run many times and check average
        counts = np.zeros(N)
        n_trials = 5000  # More trials for better statistics

        for _ in range(n_trials):
            indexes = residual_resample(weights)
            for idx in indexes:
                counts[idx] += 1

        # Expected counts
        expected_counts = weights * N * n_trials

        # Check only the top 3 particles with significant weight
        for i in range(3):
            if expected_counts[i] > 0:
                relative_error = abs(counts[i] - expected_counts[i]) / expected_counts[i]
                assert relative_error < 0.15  # Within 15% for high-weight particles

    def test_output_length_matches_input(self):
        """Test that output has same length as input."""
        for N in [10, 50, 100, 500]:
            weights = np.random.rand(N)
            weights /= np.sum(weights)

            indexes = residual_resample(weights)

            assert len(indexes) == N

    def test_all_weights_equal(self):
        """Test with all equal weights."""
        N = 50
        weights = np.ones(N) / N

        indexes = residual_resample(weights)

        # Should have roughly uniform distribution
        unique, counts = np.unique(indexes, return_counts=True)
        # Each particle should appear roughly once
        assert len(unique) >= N * 0.8  # At least 80% of particles represented

    def test_deterministic_part(self):
        """Test deterministic copying of high-weight particles."""
        weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05, 0, 0, 0, 0, 0])

        indexes = residual_resample(weights)

        # Particle 0 should appear at least int(10*0.5) = 5 times
        assert np.sum(indexes == 0) >= 5
        # Particle 1 should appear at least int(10*0.3) = 3 times
        assert np.sum(indexes == 1) >= 3


class TestStratifiedResample:
    """Tests for stratified_resample function."""

    def test_uniform_weights(self):
        """Test with uniform weights."""
        N = 100
        weights = np.ones(N) / N

        indexes = stratified_resample(weights)

        assert len(indexes) == N
        assert np.all(indexes >= 0)
        assert np.all(indexes < N)

    def test_skewed_weights(self):
        """Test with skewed weight distribution."""
        N = 100
        weights = np.random.rand(N)
        weights /= np.sum(weights)

        indexes = stratified_resample(weights)

        assert len(indexes) == N
        # High weight particles should be selected more often
        max_weight_idx = np.argmax(weights)
        assert np.sum(indexes == max_weight_idx) > 0

    def test_stratification_property(self):
        """Test that samples are stratified (between 0 and 2/N apart)."""
        np.random.seed(42)
        N = 100
        weights = np.ones(N) / N

        # Run multiple times to check property
        for _ in range(10):
            indexes = stratified_resample(weights)
            # With uniform weights, should have good coverage
            unique_indexes = len(np.unique(indexes))
            # Should have many unique particles
            assert unique_indexes >= N * 0.7

    def test_output_length_matches_input(self):
        """Test that output has same length as input."""
        for N in [10, 50, 100]:
            weights = np.random.rand(N)
            weights /= np.sum(weights)

            indexes = stratified_resample(weights)

            assert len(indexes) == N


class TestSystematicResample:
    """Tests for systematic_resample function."""

    def test_uniform_weights(self):
        """Test with uniform weights."""
        N = 100
        weights = np.ones(N) / N

        indexes = systematic_resample(weights)

        assert len(indexes) == N
        assert np.all(indexes >= 0)
        assert np.all(indexes < N)

    def test_deterministic_with_seed(self):
        """Test that systematic resampling is deterministic with same seed."""
        np.random.seed(42)
        N = 50
        weights = np.random.rand(N)
        weights /= np.sum(weights)

        np.random.seed(100)
        indexes1 = systematic_resample(weights)

        np.random.seed(100)
        indexes2 = systematic_resample(weights)

        assert np.array_equal(indexes1, indexes2)

    def test_samples_exactly_1_over_n_apart(self):
        """Test systematic property: samples are exactly 1/N apart."""
        # This is an internal property that's hard to test directly,
        # but we can verify the distribution is smooth
        np.random.seed(42)
        N = 100
        weights = np.ones(N) / N

        indexes = systematic_resample(weights)

        # With uniform weights, should have very good coverage
        unique_indexes = len(np.unique(indexes))
        assert unique_indexes >= N * 0.9  # At least 90% coverage

    def test_output_length_matches_input(self):
        """Test that output has same length as input."""
        for N in [10, 50, 100]:
            weights = np.random.rand(N)
            weights /= np.sum(weights)

            indexes = systematic_resample(weights)

            assert len(indexes) == N


class TestMultinomialResample:
    """Tests for multinomial_resample function."""

    def test_uniform_weights(self):
        """Test with uniform weights."""
        N = 100
        weights = np.ones(N) / N

        indexes = multinomial_resample(weights)

        assert len(indexes) == N
        assert np.all(indexes >= 0)
        assert np.all(indexes < N)

    def test_skewed_weights(self):
        """Test with highly skewed weights."""
        N = 100
        weights = np.random.rand(N)
        weights /= np.sum(weights)

        indexes = multinomial_resample(weights)

        assert len(indexes) == N
        # Particle with max weight should appear
        max_idx = np.argmax(weights)
        assert max_idx in indexes

    def test_output_length_matches_input(self):
        """Test that output has same length as input."""
        for N in [10, 50, 100, 500]:
            weights = np.random.rand(N)
            weights /= np.sum(weights)

            indexes = multinomial_resample(weights)

            assert len(indexes) == N

    def test_preserves_distribution_statistically(self):
        """Test that multinomial preserves weight distribution."""
        np.random.seed(42)
        N = 30
        weights = np.array([0.5, 0.3, 0.2])
        weights = np.concatenate([weights, np.zeros(N - 3)])

        counts = np.zeros(N)
        n_trials = 10000

        for _ in range(n_trials):
            indexes = multinomial_resample(weights)
            for idx in indexes:
                counts[idx] += 1

        # Expected counts
        expected_counts = weights * N * n_trials

        # Check top 3 particles
        for i in range(3):
            relative_error = abs(counts[i] - expected_counts[i]) / expected_counts[i]
            assert relative_error < 0.1  # Within 10%


class TestCrossMethodComparison:
    """Cross-comparison tests between resampling methods."""

    def test_all_methods_return_valid_indexes(self):
        """Test that all methods return valid particle indexes."""
        N = 100
        weights = np.random.rand(N)
        weights /= np.sum(weights)

        for resample_fn in [residual_resample, stratified_resample, systematic_resample, multinomial_resample]:
            indexes = resample_fn(weights)

            assert len(indexes) == N
            assert np.all(indexes >= 0)
            assert np.all(indexes < N)
            assert indexes.dtype in [np.int32, np.int64, int]

    def test_all_methods_preserve_weights_statistically(self):
        """Test that all methods preserve weight distribution statistically."""
        np.random.seed(42)
        N = 10
        # Use simple skewed distribution
        weights = np.array([0.5, 0.3, 0.2] + [0.0] * 7)

        n_trials = 5000

        for resample_fn in [residual_resample, stratified_resample, systematic_resample, multinomial_resample]:
            counts = np.zeros(N)

            for _ in range(n_trials):
                indexes = resample_fn(weights)
                for idx in indexes:
                    counts[idx] += 1

            # Expected counts
            expected_counts = weights * N * n_trials

            # Check top 3 particles only
            for i in range(3):
                if expected_counts[i] > 0:
                    relative_error = abs(counts[i] - expected_counts[i]) / expected_counts[i]
                    assert relative_error < 0.15  # Within 15%

    def test_uniform_weights_all_methods(self):
        """Test all methods with uniform weights."""
        N = 100
        weights = np.ones(N) / N

        for resample_fn in [residual_resample, stratified_resample, systematic_resample, multinomial_resample]:
            indexes = resample_fn(weights)

            # With uniform weights, should have good particle diversity
            unique_count = len(np.unique(indexes))
            assert unique_count >= N * 0.5  # At least half the particles

    def test_weights_sum_to_one_edge_case(self):
        """Test that methods handle weights that sum exactly to 1.0."""
        N = 10
        weights = np.ones(N) / N
        weights[-1] = 1.0 - np.sum(weights[:-1])  # Ensure exact sum

        for resample_fn in [residual_resample, stratified_resample, systematic_resample, multinomial_resample]:
            indexes = resample_fn(weights)
            assert len(indexes) == N

    @pytest.mark.parametrize("N", [5, 10, 50, 100])
    def test_all_methods_various_sizes(self, N):
        """Test all methods with various population sizes."""
        weights = np.random.rand(N)
        weights /= np.sum(weights)

        for resample_fn in [residual_resample, stratified_resample, systematic_resample, multinomial_resample]:
            indexes = resample_fn(weights)

            assert len(indexes) == N
            assert np.all(indexes >= 0)
            assert np.all(indexes < N)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
