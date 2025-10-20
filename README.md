# Bayesian Filters - Kalman filters and other optimal and non-optimal estimation filters in Python

<p align="center">
  <img width="100%" alt="Kalman Filter Illustration" src="https://github.com/user-attachments/assets/90b0b305-3128-4806-a72c-a061d01a854b" />
</p>

For people new to Kalman filters, they're well explained here https://www.youtube.com/watch?v=IFeCIbljreY (credit for the above image)

[![PyPI version](https://img.shields.io/pypi/v/bayesian-filters.svg)](https://pypi.org/project/bayesian-filters/)
[![Documentation Status](https://img.shields.io/badge/docs-online-brightgreen)](https://georgepearse.github.io/bayesian_filters)
[![PyPI Publishing](https://github.com/GeorgePearse/bayesian_filters/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/GeorgePearse/bayesian_filters/actions/workflows/publish-pypi.yml)
[![Latest Release](https://img.shields.io/badge/latest%20release-v1.4.5-success)](https://pypi.org/project/bayesian-filters/1.4.5/)

> **Note**: This is a personal fork of the original FilterPy library (now renamed to Bayesian Filters). The original project can be found at https://github.com/rlabbe/filterpy
>
> Maintained by George Pearse, Lead MLE at [Visia](https://www.visia.ai/)

This library provides Kalman filtering and various related optimal and non-optimal filtering software written in Python. It contains Kalman filters, Extended Kalman filters, Unscented Kalman filters, Kalman smoothers, Least Squares filters, fading memory filters, g-h filters, discrete Bayes, and more.

This is code originally developed in conjunction with the book [Kalman and Bayesian Filter in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/).

## Design Philosophy

The original author's aim is largely pedagogical - opting for clear code that matches the equations in the relevant texts on a 1-to-1 basis, even when that has a performance cost. There are places where this tradeoff is unclear - for example, writing a small set of equations using linear algebra may be clearer, but NumPy's overhead on small matrices makes it run slower than writing each equation out by hand.

All computations use NumPy and SciPy.

## Documentation

Documentation is available at https://georgepearse.github.io/bayesian_filters

## Installation

```bash
uv pip install bayesian-filters
```


## Basic Usage

Full documentation is at https://georgepearse.github.io/bayesian_filters

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

The book [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/) uses this library and is the best place to learn about Kalman filtering and/or this library.

## Requirements

This library requires:
- Python 3.9+
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

- **[Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)** - Free online book with Jupyter notebooks teaching Kalman filtering and Bayesian statistics. Written by the original FilterPy author, this is the companion book to this library and provides excellent intuitive explanations with interactive examples.
- **[Kalman Filter Background](https://kalmanfilter.net/background.html)** - Comprehensive background and theory on Kalman filtering

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/GeorgePearse/bayesian_filters/blob/master/LICENSE)

The MIT License (MIT)

Copyright (c) 2015 Roger R. Labbe Jr

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
