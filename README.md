# Bayesian Filters - Kalman filters and other optimal and non-optimal estimation filters in Python

<p align="center">
  <img width="100%" alt="Kalman Filter Illustration" src="https://github.com/user-attachments/assets/90b0b305-3128-4806-a72c-a061d01a854b" />
</p>

For people new to Kalman filters, they're well explained here https://www.youtube.com/watch?v=IFeCIbljreY (credit for the above image)

[![PyPI version](https://img.shields.io/pypi/v/bayesian-filters.svg)](https://pypi.org/project/bayesian-filters/)
[![Documentation Status](https://img.shields.io/badge/docs-online-brightgreen)](https://georgepearse.github.io/bayesian_filters)
[![PyPI Publishing](https://github.com/GeorgePearse/bayesian_filters/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/GeorgePearse/bayesian_filters/actions/workflows/publish-pypi.yml)
[![Latest Release](https://img.shields.io/badge/latest%20release-v1.4.5-success)](https://pypi.org/project/bayesian-filters/1.4.5/)
[![codecov](https://codecov.io/gh/GeorgePearse/bayesian_filters/branch/master/graph/badge.svg)](https://codecov.io/gh/GeorgePearse/bayesian_filters)

Maintained by George Pearse, Lead MLE at [Visia](https://www.visia.ai/)

This is the fork of the original and excellent filterpy (https://github.com/rlabbe/filterpy), only forked because of the pull requests left without response for essential upgrades to packaging to support python 3.12+ etc. etc. 99% of the functionality remains identical, but I'm adding testing and typing etc. so that the repo is ready for lower cost upgrades.

This library provides Kalman filtering and various related optimal and non-optimal filtering software written in Python. It contains Kalman filters, Extended Kalman filters, Unscented Kalman filters, Kalman smoothers, Least Squares filters, fading memory filters, g-h filters, discrete Bayes, and more.

This is a comprehensive implementation of Kalman filters and related estimation algorithms in Python.

All computations use NumPy and SciPy.

## Documentation

Documentation is available at https://georgepearse.github.io/bayesian_filters

## Installation

```bash
uv pip install bayesian-filters
```


## Basic Usage

### Import the filters

```python
import numpy as np
import bayesian_filters as bf

# Or import specific modules
from bayesian_filters.kalman import KalmanFilter
from bayesian_filters.common import Q_discrete_white_noise
```

### Create the filter

```python
my_filter = KalmanFilter(dim_x=2, dim_z=1)
```

### Initialize the filter's matrices

```python
my_filter.x = np.array([[2.],
                [0.]])       # initial state (location and velocity)

my_filter.F = np.array([[1.,1.],
                [0.,1.]])    # state transition matrix

my_filter.H = np.array([[1.,0.]])    # Measurement function
my_filter.P *= 1000.                 # covariance matrix
my_filter.R = 5                      # state uncertainty
my_filter.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.1) # process uncertainty
```

### Run the filter

```python
while True:
    my_filter.predict()
    my_filter.update(get_some_measurement())

    # do something with the output
    x = my_filter.x
    do_something_amazing(x)
```

## Library Structure

The library is broken up into subdirectories:
- `gh` - g-h filters
- `kalman` - Kalman filters (KF, EKF, UKF, etc.)
- `memory` - fading memory filters
- `leastsq` - least squares filters
- `discrete_bayes` - discrete Bayes filters
- `stats` - statistical functions
- `common` - common helper functions

Each subdirectory contains Python files relating to that form of filter. The functions and methods contain comprehensive docstrings.

## Requirements

This library requires:
- Python 3.10+
- NumPy
- SciPy
- Matplotlib

## Testing

All tests are written to work with pytest. Just run:

```bash
pytest
```

The tests include both unit tests and visual verification. Visual plots are often the best way to see how filters are working, as it's easy for a filter to perform within theoretical limits yet be 'off' in some way.

## References

### Books

The original author uses three main reference texts:

1. **Paul Zarchan's 'Fundamentals of Kalman Filtering: A Practical Approach'** - Excellent for practical applications rather than theoretical thesis work.

2. **Eli Brookner's 'Tracking and Kalman Filtering Made Easy'** - An astonishingly good book with its first chapter readable by laypeople. Brookner starts from the g-h filter and shows how all other filters derive from it, including Kalman filters, least squares, and fading memory filters. Also focuses on practical issues like track initialization, noise detection, and tracking multiple objects.

3. **Bar-Shalom's 'Estimation with Applications to Tracking and Navigation'** - More mathematical than the previous two. Recommended after gaining some background in control theory or optimal estimation. Every sentence is crystal clear with precise language, and abstract mathematical statements are followed with practical explanations.

### Online Resources

- **[Kalman Filter Background](https://kalmanfilter.net/background.html)** - Comprehensive background and theory on Kalman filtering

## Tools and Projects Using Kalman Filters

### Repositories Using Bayesian Filters

- **[Gavin-Furtado/Kalman-Filter-Simulator](https://github.com/Gavin-Furtado/Kalman-Filter-Simulator)** - Python project simulating sensor tracking with state estimation
- **[sparshgarg23/object-detection-and-tracking](https://github.com/sparshgarg23/object-detection-and-tracking)** - Object detection pipeline with OpenCV and FilterPy-based Kalman filters
- **[Norfair](https://github.com/tryolabs/norfair)** - Lightweight Python library for real-time multi-object tracking using Kalman filters

### Collections and Resources

- **[GitHub Topic: filterpy](https://github.com/topics/filterpy)** - Projects tagged with "filterpy" covering robotics, tracking, and sensor fusion applications
- **[mintisan/awesome-kalman-filter](https://github.com/mintisan/awesome-kalman-filter)** - Curated list of Kalman filter implementations, tutorials, and resources

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/GeorgePearse/bayesian_filters/blob/master/LICENSE)

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
