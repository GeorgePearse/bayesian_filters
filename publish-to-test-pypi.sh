#!/bin/bash
set -e

echo "Publishing bayesian-filters to TestPyPI"
echo "========================================"

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
read -p "Do you want to publish to TestPyPI? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Publish cancelled."
    exit 0
fi

# Check if TEST_PYPI_TOKEN is set
if [ -z "$TEST_PYPI_TOKEN" ]; then
    echo "Error: TEST_PYPI_TOKEN environment variable not set"
    echo "Please set it with: export TEST_PYPI_TOKEN=your-token-here"
    exit 1
fi

# Publish to TestPyPI using uv
echo "Publishing to TestPyPI..."
uv publish --publish-url https://test.pypi.org/legacy/ --token "$TEST_PYPI_TOKEN"

echo ""
echo "Successfully published to TestPyPI!"
echo "Install with: pip install --index-url https://test.pypi.org/simple/ bayesian-filters"
