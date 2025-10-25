# Bayesian Filters

**Kalman filtering and optimal estimation library**

This library provides Kalman filtering and various related optimal and non-optimal filtering software written in Python. It contains:

- **Kalman filters**: Standard, Extended, Unscented, Ensemble, Information, Square Root
- **Smoothers**: Kalman smoothers, Fixed Lag smoothers
- **Other filters**: H-Infinity, Fading memory, g-h filters, Least Squares
- **Bayesian methods**: Discrete Bayes, Monte Carlo
- **Multi-model**: IMM Estimator, MMAE Filter Bank

## Quick Install

```bash
pip install bayesian-filters
```

## Quick Example

```python
import numpy as np
from bayesian_filters.kalman import KalmanFilter
from bayesian_filters.common import Q_discrete_white_noise

# Create filter
my_filter = KalmanFilter(dim_x=2, dim_z=1)

# Initialize matrices
my_filter.x = np.array([[2.], [0.]])  # initial state (position and velocity)
my_filter.F = np.array([[1., 1.], [0., 1.]])  # state transition matrix
my_filter.H = np.array([[1., 0.]])  # measurement function
my_filter.P *= 1000.  # covariance matrix
my_filter.R = 5  # measurement uncertainty
my_filter.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.1)  # process uncertainty

# Run filter
while True:
    my_filter.predict()
    my_filter.update(get_measurement())

    # Use the filtered output
    x = my_filter.x
    do_something_with(x)
```

## Features

### Comprehensive Filter Collection

This library implements a wide range of filtering algorithms:

- **Standard Kalman Filter**: The classic discrete Kalman filter
- **Extended Kalman Filter (EKF)**: For nonlinear systems using linearization
- **Unscented Kalman Filter (UKF)**: For nonlinear systems using unscented transform
- **Ensemble Kalman Filter (EnKF)**: Monte Carlo-based approach
- **Square Root Filter**: Numerically stable variant
- **Information Filter**: Inverse covariance form of Kalman filter

### Clear, Pedagogical Code

The code is written to match equations from textbooks on a 1-to-1 basis, making it easy to learn and understand. This library was developed in conjunction with the book [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/).

### Well Tested

All filters include comprehensive test suites and are used in production systems.

## License

MIT License - see [LICENSE](https://github.com/GeorgePearse/filterpy/blob/master/LICENSE) for details.

## Documentation Sections

- **[Getting Started](getting-started/installation.md)**: Installation and quick start guide
- **[Filters](filters/kalman-filter.md)**: Detailed documentation on each filter type
- **[Algorithms](algorithms/gh-filter.md)**: Other estimation algorithms
- **[API Reference](api/kalman.md)**: Complete API documentation
- **[Examples](examples.md)**: Working examples and tutorials

## Resources

- **Book**: [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/)
- **GitHub**: [https://github.com/GeorgePearse/filterpy](https://github.com/GeorgePearse/filterpy)
- **PyPI**: [https://pypi.org/project/bayesian-filters/](https://pypi.org/project/bayesian-filters/)
