# Contributing to Tennis Match Predictor

Thank you for your interest in contributing to the Tennis Match Predictor project! This document provides guidelines and instructions for contributing.

## 🌟 Ways to Contribute

- 🐛 **Bug Reports**: Report bugs through GitHub Issues
- ✨ **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve or add documentation
- 🧪 **Testing**: Add or improve test coverage
- 💻 **Code**: Submit pull requests with bug fixes or new features

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- `uv` package manager
- Git
- Docker (optional)

### Setup Development Environment

1. **Fork and clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/tennis-match-predictor.git
cd tennis-match-predictor
```

2. **Install dependencies**
```bash
uv sync
```

3. **Install pre-commit hooks** (optional)
```bash
uv run pre-commit install
```

4. **Run tests to verify setup**
```bash
uv run pytest
```

## 📋 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, readable code
- Follow the existing code style
- Add tests for new features
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run tests
uv run pytest

# Run linting
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/

# Check coverage
uv run pytest --cov=match_predictor --cov-report=html
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add amazing feature"
```

**Commit Message Convention:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `style:` Code style changes (formatting, etc.)
- `chore:` Maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## 📝 Code Style

### Python Style Guide

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check code style
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/
```

### Docstring Style

We use Google-style docstrings:

```python
def predict_match(player1: str, player2: str) -> dict:
    """
    Predict the outcome of a tennis match.

    Args:
        player1: Name of the first player
        player2: Name of the second player

    Returns:
        Dictionary containing prediction results with win probabilities

    Raises:
        ValueError: If player names are invalid
    """
    pass
```

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test names: `test_feature_engineering_adds_stats()`
- Use fixtures for common test data
- Aim for high test coverage (>80%)

### Test Structure

```python
import pytest

@pytest.fixture
def sample_data():
    """Fixture providing sample test data."""
    return {"player": "Test Player"}

def test_feature_description(sample_data):
    """Test description explaining what is being tested."""
    # Arrange
    expected = "expected_value"
    
    # Act
    result = some_function(sample_data)
    
    # Assert
    assert result == expected
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_api.py

# Run with coverage
uv run pytest --cov=match_predictor

# Run with verbose output
uv run pytest -v
```

## 📚 Documentation

### Code Documentation

- Add docstrings to all public functions, classes, and methods
- Use type hints for all function parameters and return values
- Keep docstrings up-to-date with code changes

### MkDocs Documentation

Documentation is built with MkDocs and mkdocstrings:

```bash
# Serve documentation locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build

# Deploy to GitHub Pages
uv run mkdocs gh-deploy
```

### Adding New Documentation Pages

1. Create a new `.md` file in the `docs/` directory
2. Add the page to `mkdocs.yml` in the `nav` section
3. Use clear headings and examples
4. Include code snippets where appropriate

## 🐛 Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: Python version, OS, etc.
- **Screenshots**: If applicable

## ✨ Feature Requests

When suggesting features, please include:

- **Use Case**: Why is this feature needed?
- **Description**: Detailed description of the feature
- **Examples**: Examples of how it would work
- **Alternatives**: Any alternative solutions you've considered

## 📦 Pull Request Process

1. **Ensure tests pass**: All tests must pass before merging
2. **Update documentation**: Add or update relevant documentation
3. **Add tests**: Include tests for new features
4. **Review changes**: Request review from maintainers
5. **Address feedback**: Make requested changes
6. **Squash commits**: Squash commits before merging (if requested)

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] Tests added for new features
- [ ] All tests passing
```

## 🏗️ Project Structure

```
src/match_predictor/
├── api/            # FastAPI service
├── app/            # Streamlit app
├── ml_pipeline/    # ML training and monitoring
├── data/           # Data loading
├── model/          # Model classes
└── utils/          # Utilities
```

## 🎯 Focus Areas

We're particularly interested in contributions related to:

- 🧪 **Testing**: Improving test coverage
- 📊 **Features**: New player statistics and features
- 🎨 **UI/UX**: Streamlit app improvements
- 🚀 **Performance**: Optimization and efficiency
- 📝 **Documentation**: Clearer explanations and examples
- 🔍 **Monitoring**: Better drift detection and alerting

## ❓ Questions?

If you have questions about contributing:

1. Check existing [Issues](https://github.com/FedericoRaimondi/tennis-match-predictor/issues)
2. Create a new [Discussion](https://github.com/FedericoRaimondi/tennis-match-predictor/discussions)
3. Reach out to the maintainers

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing! 🎾**
