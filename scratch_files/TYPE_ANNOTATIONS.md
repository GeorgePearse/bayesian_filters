# Type Annotations and Zuban Configuration

This document describes how to gradually add type annotations to the bayesian-filters codebase using Zuban as the type checker.

## Overview

We're adopting type annotations incrementally to improve code quality and catch bugs early. The strategy is:

1. **Start with all files ignored** - Zuban won't check anything initially
2. **Enable modules one at a time** - As a module gets type annotations, enable it in the mypy config
3. **Gradually increase strictness** - Start permissive, tighten rules as coverage improves

## Current Status

All files in `bayesian_filters/` are currently ignored by Zuban. The configuration allows untyped definitions and functions without type annotations.

### Configuration (in `pyproject.toml`)

```toml
[tool.mypy]
# Ignore all files by default
ignore_patterns = ["^bayesian_filters/.*"]

# Permissive settings (will tighten as we add annotations)
allow_untyped_defs = true
allow_incomplete_defs = true
ignore_missing_imports = true
allow_untyped_globals = true
warn_unused_ignores = false
```

## How to Enable Type Checking for a Module

### Step 1: Add Type Annotations

Add type hints to function signatures and variable declarations in your module. For example:

```python
def kinematic_state_transition(
    dim: int,
    order: int,
    dt: float,
) -> np.ndarray:
    """Generate kinematic state transition matrix."""
    ...
```

### Step 2: Update pyproject.toml

Remove the ignore pattern for your module in the `[tool.mypy]` section:

```toml
[tool.mypy]
# Before
ignore_patterns = ["^bayesian_filters/.*"]

# After - module is now checked!
ignore_patterns = [
    "^bayesian_filters/.*",
    # Add exceptions when ready:
    "!^bayesian_filters/common/kinematic.py",  # Now type-checked
]
```

Or more simply, just remove it from the ignore list if all other files are excluded.

### Step 3: Run Zuban Locally

Check your module for type errors:

```bash
# Check a specific file
uv run zuban check bayesian_filters/common/kinematic.py

# Check the whole project
uv run zuban check .
```

### Step 4: Fix Type Errors

Address any type checking errors that Zuban reports. Use `# type: ignore` comments sparingly for legitimate cases.

### Step 5: Commit

Update the configuration and commit your typed module.

## Zuban Commands

### Check for type errors

```bash
# Check entire project
uv run zuban check .

# Check specific file
uv run zuban check bayesian_filters/kalman/kalman_filter.py

# Check with specific configuration
uv run zuban check --config-file pyproject.toml .
```

### Interactive mode (LSP server)

For IDE integration:

```bash
uv run zuban server
```

### Mypy-compatible mode

```bash
uv run zuban mypy ...
```

## Type Annotation Guidelines

### Import typing utilities

```python
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
```

### Array types

Use numpy's typing module for array annotations:

```python
def filter_step(
    z: NDArray[np.float64],
    H: NDArray[np.float64],
    R: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Process measurement update."""
    ...
```

### Optional types

```python
def __init__(
    self,
    dim_x: int,
    dim_z: int,
    cov: Optional[NDArray] = None,
) -> None:
    ...
```

### Union types

```python
def process(
    value: Union[int, float],
    state: Union[np.ndarray, list[float]],
) -> float:
    ...
```

### Callbacks

```python
from typing import Callable

def integrate(
    f: Callable[[float, NDArray], NDArray],
    y0: NDArray,
    t: NDArray,
) -> NDArray:
    """Integrate differential equation."""
    ...
```

## Gradual Strictness

As we increase type annotation coverage, we can make the mypy configuration stricter:

### Phase 1: Current (Very Permissive)
- ✅ Allow untyped function definitions
- ✅ Allow incomplete type hints
- ✅ Allow missing imports
- ✅ Allow untyped globals

### Phase 2: Moderate (When ~50% of code typed)
```toml
allow_untyped_defs = false  # Require type hints on all functions
allow_incomplete_defs = true  # Still allow some flexibility
warn_unused_ignores = true  # Start cleaning up `# type: ignore`
```

### Phase 3: Strict (When ~90% of code typed)
```toml
allow_incomplete_defs = false  # Require complete type coverage
disallow_any_generics = true  # Use specific types, not `Any`
warn_return_any = true  # Flag functions returning `Any`
```

### Phase 4: Maximum (Full coverage)
```toml
strict = true  # Enable all strict checks
```

## Example: Typing a Module

Here's an example of taking `bayesian_filters/common/kinematic.py` from untyped to typed:

1. **Before**: No type annotations
```python
def Q_discrete_white_noise(dim, dt, var, block_size=1, order_by_dim=True):
    if dim < 1:
        raise ValueError('dim must be >= 1')
    ...
```

2. **After**: With type annotations
```python
def Q_discrete_white_noise(
    dim: int,
    dt: float,
    var: float,
    block_size: int = 1,
    order_by_dim: bool = True,
) -> NDArray[np.float64]:
    """Generate discrete-time process noise covariance matrix.

    Parameters
    ----------
    dim : int
        Dimension of state vector
    dt : float
        Time step
    var : float
        Noise variance
    block_size : int, optional
        Size of state vector blocks, by default 1
    order_by_dim : bool, optional
        If True, orders by dimension; if False, by derivative order

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix of shape (dim*block_size, dim*block_size)
    """
    if dim < 1:
        raise ValueError('dim must be >= 1')
    ...
```

3. **Enable in pyproject.toml**:
```toml
[tool.mypy]
ignore_patterns = [
    "^bayesian_filters/.*",
    "!^bayesian_filters/common/kinematic.py",  # Now checked!
]
```

4. **Run Zuban**:
```bash
uv run zuban check bayesian_filters/common/kinematic.py
```

5. **Fix any errors and commit**

## Modules Enabled for Type Checking

Currently enabled (checked by Zuban):
- (None - all files ignored initially)

## Common Type Checking Patterns

### Skip type checking for a line

```python
result = some_untyped_function()  # type: ignore
```

### Skip type checking for a function

```python
def legacy_function():  # type: ignore
    # Type errors here are ignored
    return untyped_result * something_else
```

### Use Any when unavoidable

```python
from typing import Any

def flexible_function(value: Any) -> Any:
    """Function that works with any type."""
    return value
```

## Resources

- [Zuban Documentation](https://github.com/fruits-lab/Rust-Python-Type-Checker)
- [Python typing module](https://docs.python.org/3/library/typing.html)
- [Numpy typing](https://numpy.org/doc/stable/reference/typing.html)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
