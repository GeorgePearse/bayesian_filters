# PyPI Publishing Setup

This document describes how to set up PyPI publishing for the bayesian-filters package.

## Prerequisites

You need a PyPI API token. The token has already been generated for the `bayesian-filters` package.

## Setup Steps

### 1. Add PyPI API Token to GitHub Secrets

1. Go to the repository settings: https://github.com/GeorgePearse/bayesian_filters/settings/secrets/actions
2. Click "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Value: Paste the PyPI API token (starts with `pypi-AgEI...`)
5. Click "Add secret"

### 2. Configure PyPI Environment (Optional but Recommended)

For additional security, create a deployment environment:

1. Go to: https://github.com/GeorgePearse/bayesian_filters/settings/environments
2. Click "New environment"
3. Name it: `pypi`
4. Add protection rules if desired (e.g., require approval before publishing)
5. Save

The workflow is already configured to use this environment.

## How to Publish

### Option 1: Tag-based Publishing (Recommended)

When you want to publish a new version:

1. Update the version in `bayesian_filters/__init__.py`:
   ```python
   __version__ = "1.4.6"  # or whatever version
   ```

2. Commit the change:
   ```bash
   git add bayesian_filters/__init__.py
   git commit -m "chore: bump version to 1.4.6"
   git push
   ```

3. Create and push a git tag:
   ```bash
   git tag v1.4.6
   git push origin v1.4.6
   ```

4. The GitHub Action will automatically:
   - Build the package
   - Publish to PyPI
   - Create a GitHub release

### Option 2: Manual Publishing

1. Go to: https://github.com/GeorgePearse/bayesian_filters/actions/workflows/publish-pypi.yml
2. Click "Run workflow"
3. Select the branch (usually `master`)
4. Click "Run workflow"

## Verification

After publishing, verify the package is available:

- PyPI page: https://pypi.org/project/bayesian-filters/
- Install test: `pip install bayesian-filters==1.4.6`

## Current Version

Current version: `1.4.5` (in `bayesian_filters/__init__.py`)

## Notes

- The workflow uses `uv build` to create both wheel and source distributions
- The PyPI token is scoped to only the `bayesian-filters` package
- The token is stored securely as a GitHub secret and never exposed in logs
