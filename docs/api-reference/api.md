# API Module API Reference

The API module provides a FastAPI-based REST service for tennis match predictions.

## Overview

The API service exposes endpoints for:

- Health checks and service information
- **Enhanced match outcome prediction with dynamic feature extraction**
- Latest match statistics retrieval

## Prediction Endpoint

### Enhanced Feature Extraction

The `/predict_winner` endpoint has been enhanced to:

1. **Retrieve the champion model** from MLflow model registry
2. **Extract input features dynamically** using:
   - Player names (player1, player2)
   - Tournament name
   - Latest player statistics (from saved player stats file)
   - Tournament information (surface type, location, level)
3. **Construct inference-ready dataset** with proper feature engineering
4. **Perform prediction** and return:
   - Win probabilities for both players
   - Predicted winner
   - **Latest 5 match statistics** for context and display

### Feature Consistency

The prediction endpoint uses the same feature engineering pipeline as model training to ensure:
- Consistent feature schema between training and inference
- Proper handling of categorical variables
- Correct application of derived features (e.g., service percentages, rank differences)

## API Reference

::: match_predictor.api.main
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
