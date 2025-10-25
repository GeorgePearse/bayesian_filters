# Publishing to PyPI

This guide explains how to publish `bayesian-filters` to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts at:
   - [TestPyPI](https://test.pypi.org/account/register/) (for testing)
   - [PyPI](https://pypi.org/account/register/) (for production)

2. **API Tokens**: Generate tokens at:
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - PyPI: https://pypi.org/manage/account/token/

3. **Set Environment Variables**:
   ```bash
   export TEST_PYPI_TOKEN="pypi-your-test-token-here"
   export PYPI_TOKEN="pypi-your-production-token-here"
   ```

## Publishing Workflow

### 1. Test on TestPyPI First

Always test your package on TestPyPI before publishing to production PyPI:

```bash
./scripts/publish-to-test-pypi.sh
```

This will:
- Clean previous builds
- Build the package with `uv build`
- Ask for confirmation
- Publish to TestPyPI

### 2. Test Installation from TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ bayesian-filters
```

Test that the package works:
```python
import bayesian_filters as bf
from bayesian_filters.kalman import KalmanFilter
print(bf.__version__)
```

### 3. Publish to Production PyPI

Once you've verified the package works from TestPyPI:

```bash
./scripts/publish-to-pypi.sh
```

This will:
- Clean previous builds
- Build the package with `uv build`
- Ask for confirmation
- Publish to PyPI

### 4. Verify Installation

```bash
pip install bayesian-filters
```

## Manual Publishing (Alternative)

If you prefer to publish manually:

```bash
# Build
uv build

# Publish to TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/ --token "$TEST_PYPI_TOKEN"

# Publish to PyPI
uv publish --token "$PYPI_TOKEN"
```

## Version Management

Update the version in `bayesian_filters/__init__.py`:
```python
__version__ = "1.4.6"  # Update this
```

And in `pyproject.toml`:
```toml
version = "1.4.6"  # Update this
```

## Package Information

- **PyPI Package Name**: `bayesian-filters`
- **Python Module Name**: `bayesian_filters`
- **Import Statement**: `import bayesian_filters as bf`
- **Repository**: https://github.com/GeorgePearse/filterpy
- **Original Project**: https://github.com/rlabbe/filterpy

## Troubleshooting

**Build fails**: Ensure you're in the project root with `pyproject.toml`

**Token errors**: Make sure your token starts with `pypi-` and is properly exported

**Version conflicts**: PyPI doesn't allow re-uploading the same version. Increment the version number.

**Import errors after install**: Make sure you're importing `bayesian_filters` not `filterpy`
