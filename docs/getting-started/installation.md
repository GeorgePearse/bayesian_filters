# Installation

## Using pip (Recommended)

The easiest way to install Bayesian Filters is using pip:

```bash
pip install bayesian-filters
```

## Using uv

If you use [uv](https://github.com/astral-sh/uv) for Python package management:

```bash
uv pip install bayesian-filters
```

## Using Anaconda

If you use Anaconda, you can install from the conda-forge channel:

```bash
# Add conda-forge channel if you haven't already
conda config --add channels conda-forge

# Install the package
conda install filterpy
```

Note: The conda package is still named `filterpy` for backward compatibility.

## From Source

To install the latest development version from GitHub:

```bash
pip install git+https://github.com/GeorgePearse/filterpy.git
```

Or clone and install locally:

```bash
git clone https://github.com/GeorgePearse/filterpy.git
cd filterpy
pip install -e .
```

## Requirements

- Python >= 3.6
- NumPy
- SciPy
- Matplotlib

These dependencies will be installed automatically when you install the package.

## Verifying Installation

To verify the installation, try importing the library:

```python
import bayesian_filters
from bayesian_filters.kalman import KalmanFilter

print(f"Bayesian Filters version: {bayesian_filters.__version__}")
```

If this runs without errors, you're ready to go!

## Next Steps

Continue to the [Quick Start](quick-start.md) guide to learn how to use the library.
