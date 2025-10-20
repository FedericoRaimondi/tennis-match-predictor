# Data Module API Reference

The data module provides utilities for loading, processing, and saving tennis match data.

## Overview

The `DataLoader` class is the main interface for working with tennis match data. It provides methods to:

- Load match data from GitHub repositories
- Generate player statistics with rolling averages and ELO ratings
- Extract tournament information
- Prepare ML-ready datasets
- **Save latest player statistics for quick inference access**

## Key Features

### Player Statistics

The `get_player_stats()` method calculates comprehensive player statistics including:
- Rolling averages over 3, 5, and 10 match windows
- Service statistics (aces, double faults, first serve percentage)
- Break point statistics
- ELO ratings

### Latest Stats Persistence

The new `save_latest_player_stats()` method extracts and saves the most recent statistics for each player, enabling:
- Fast access during model inference
- Consistent feature generation between training and prediction
- Support for both CSV and Parquet formats

## API Reference

::: match_predictor.data.data_loader
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
