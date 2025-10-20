# Contributing

Thank you for considering contributing to Bayesian Filters!

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/bayesian_filters.git
   cd bayesian_filters
   ```

3. Install development dependencies using uv:
   ```bash
   uv sync --extra dev
   ```

4. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Making Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests
3. Run the test suite:
   ```bash
   uv run pytest
   ```

4. Run pre-commit checks:
   ```bash
   uv run pre-commit run --all-files
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

## Submitting a Pull Request

1. Push your branch to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a Pull Request on GitHub
3. Ensure all CI checks pass
4. Wait for review

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write comprehensive docstrings (NumPy style)
- Keep code clear and readable over clever optimizations

## Testing

- Add tests for new functionality
- Ensure all tests pass before submitting
- Aim for high test coverage

## Documentation

- Update documentation for any changed functionality
- Add docstrings to new functions/classes
- Update relevant markdown files in `docs-mkdocs/`

## Questions?

Feel free to open an issue on GitHub if you have questions about contributing!
