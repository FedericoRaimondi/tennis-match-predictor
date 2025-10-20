# Streamlit App

The Streamlit app provides an interactive user interface for making match predictions and visualizing tennis statistics.

## Features

### 1. Match Prediction

- Select two players from the dropdown menus
- Choose a tournament
- Get real-time win probability predictions
- View head-to-head statistics

### 2. Recent Matches

- View the last 5 matches for each player
- See match statistics and results
- Filter by player or tournament

### 3. Visualizations

Interactive charts powered by Plotly:

- Win probability comparison
- Player performance trends
- Head-to-head history
- Tournament statistics

## Using the App

### Starting the App

```bash
# Local development
uv run streamlit run src/match_predictor/app/main.py

# Or use the helper script
./scripts/run_streamlit.sh
```

### Making a Prediction

1. Navigate to http://localhost:8501
2. Select **Player 1** from the first dropdown
3. Select **Player 2** from the second dropdown
4. Choose a **Tournament** (optional)
5. Click **Predict Winner**
6. View the prediction results and statistics

## Configuration

The Streamlit app uses the same configuration as the rest of the system:

- Data source settings from `config/data_config.yaml`
- Model settings from `config/model_config.yaml`

## Customization

You can customize the app by modifying `src/match_predictor/app/main.py`:

```python
# Change the page title
st.set_page_config(
    page_title="Custom Title",
    page_icon="🎾",
    layout="wide"
)

# Add custom styling
st.markdown("""
<style>
.main { background-color: #f0f0f0; }
</style>
""", unsafe_allow_html=True)
```

## Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Visit https://share.streamlit.io
3. Connect your GitHub repository
4. Deploy the app

### Docker

```bash
docker-compose up streamlit
```

## Troubleshooting

### Port Already in Use

If port 8501 is already in use:

```bash
uv run streamlit run src/match_predictor/app/main.py --server.port 8502
```

### Connection Issues

Ensure the FastAPI backend is running:

```bash
uv run uvicorn match_predictor.api.main:app --reload
```
