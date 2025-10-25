#!/bin/bash
set -e

echo "Publishing bayesian-filters to PyPI"
echo "===================================="

# Clean any previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build the package using uv
echo "Building package with uv..."
uv build

# Check the built distribution
echo ""
echo "Built distributions:"
ls -lh dist/

# Prompt for confirmation
echo ""
read -p "Do you want to publish to PyPI? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Publish cancelled."
    exit 0
fi

# Check if PYPI_TOKEN is set
if [ -z "$PYPI_TOKEN" ]; then
    echo "Error: PYPI_TOKEN environment variable not set"
    echo "Please set it with: export PYPI_TOKEN=your-token-here"
    exit 1
fi

# Publish to PyPI using uv
echo "Publishing to PyPI..."
uv publish --token "$PYPI_TOKEN"

echo ""
echo "Successfully published to PyPI!"
echo "Install with: pip install bayesian-filters"
