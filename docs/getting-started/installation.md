# Installation

This guide will help you set up the Tennis Match Predictor on your local machine.

## Prerequisites

- Python 3.13 or higher
- `uv` package manager
- Git
- Docker and Docker Compose (optional, for containerized deployment)

## Installing uv

If you don't have `uv` installed:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

## Clone the Repository

```bash
git clone https://github.com/FedericoRaimondi/tennis-match-predictor.git
cd tennis-match-predictor
```

## Install Dependencies

Using `uv` (recommended):

```bash
# Install project and dependencies
uv sync

# Install with development dependencies
uv sync --all-extras
```

## Verify Installation

```bash
# Run tests
uv run pytest

# Check API
uv run uvicorn match_predictor.api.main:app --reload

# Check Streamlit app
uv run streamlit run src/match_predictor/app/main.py
```

## Docker Installation

If you prefer using Docker:

```bash
# Build all services
docker-compose build

# Start services
docker-compose up

# Run in detached mode
docker-compose up -d
```

## Troubleshooting

### Issue: uv sync fails

**Solution**: Ensure you have Python 3.13+ installed:
```bash
python --version
```

### Issue: Import errors

**Solution**: Make sure you're using `uv run` to execute scripts:
```bash
uv run python your_script.py
```

### Issue: Data files not found

**Solution**: Ensure data files are in the `data/` directory:
```bash
ls data/
# Should show: matches_results.pkl, tournament_info.pkl
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Configuration](configuration.md)
