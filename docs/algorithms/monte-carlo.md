# Monte Carlo Methods

Monte Carlo methods use random sampling to solve problems that might be deterministic in principle.

## Overview

In the context of filtering, Monte Carlo methods are primarily used for:

- **Particle Filters** - Represent the posterior distribution using a set of particles
- **Resampling** - Techniques for selecting particles based on their weights
- **Importance Sampling** - Drawing samples from a proposal distribution

## Resampling Methods

The library provides several resampling algorithms:

- **Multinomial Resampling**
- **Residual Resampling**
- **Stratified Resampling**
- **Systematic Resampling**

## API Reference

For detailed API documentation, see the [Monte Carlo API reference](../api/monte-carlo.md).

## Further Reading

For comprehensive examples and theory, see the companion book:
[Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/)
