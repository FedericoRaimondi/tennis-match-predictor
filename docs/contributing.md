# Contributing to Tennis Match Predictor

We welcome contributions! This guide will help you get started.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch
4. Make your changes
5. Submit a pull request

## Development Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/tennis-match-predictor.git
cd tennis-match-predictor

# Install dependencies
uv sync

# Run tests
uv run pytest
```

## Code Style

We use Ruff for linting and formatting:

```bash
# Lint code
uv run ruff check src/

# Format code
uv run ruff format src/
```

## Testing

All code should have tests:

```bash
# Run all tests
uv run pytest

# Run specific tests
uv run pytest tests/test_api.py

# Run with coverage
uv run pytest --cov=match_predictor --cov-report=html
```

## Pull Request Process

1. Update tests for your changes
2. Ensure all tests pass
3. Update documentation if needed
4. Submit PR with clear description
5. Address review feedback

## Code of Conduct

Be respectful and constructive in all interactions.

For more details, see [CONTRIBUTING.md](https://github.com/FedericoRaimondi/tennis-match-predictor/blob/main/CONTRIBUTING.md).
