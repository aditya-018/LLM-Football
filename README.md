# LLM Football Analytics Project

## Project Overview
This project is a football analytics proof-of-concept that combines StatsBomb data ingestion, xG modeling, tactical clustering, visual analytics, and local LLM-powered narrative generation in a single end-to-end pipeline.

## What was built

### 1. Data ingestion and processing
- Ingests StatsBomb event JSON files from `data/statsbomb`
- Parses match and event data to extract shot events, team actions, and match metadata
- Supports reusable event discovery and local storage for downstream analysis

### 2. Expected goals (xG) modeling
- Implements shot-level feature engineering in `analytics/xgboost_model.py`
- Trains an xG model using `xgboost` and `scikit-learn`
- Handles categorical preprocessing with `OneHotEncoder` and numeric shot features
- Saves and loads the trained xG model for repeatable predictions
- Produces match-level xG summaries for each team and shot event

### 3. Tactical clustering
- Builds team tactical profiles from event signature counts
- Applies K-Means clustering to group teams by playing style
- Uses cluster output for tactical comparison and profiling

### 4. Streamlit analytics dashboard
- Provides an interactive UI in `streamlit_app.py`
- Allows team and season selection from the local dataset
- Displays team and match summaries, shot maps, and analytical charts
- Integrates xG summaries, tactical cluster views, and narrative report generation

### 5. Local LLM narrative generation
- Adds `llm/ollama_report.py` for prompt creation and Ollama integration
- Detects local Ollama availability and prepares narrative report prompts
- Enables natural-language coaching/analysis reports using local LLM models such as `llama3`

### 6. Verification and validation
- Adds `verify_project.py` as an end-to-end smoke test script
- Validates the full pipeline including:
  - StatsBomb event file discovery
  - shot event collection
  - xG model training and loading
  - match summary generation
  - team profile building
  - clustering
  - LLM prompt generation
  - Ollama availability

## Key files
- `streamlit_app.py` — interactive analytics dashboard and UI
- `analytics/xgboost_model.py` — xG feature engineering, training, and match summary logic
- `llm/ollama_report.py` — LLM report prompt generation and Ollama integration
- `verify_project.py` — end-to-end smoke test
- `requirements.txt` — dependency list

## Current status
- Core analytics pipeline implemented
- xG model and team clustering validated
- Streamlit interface built
- LLM prompt generation ready for local model execution
- End-to-end verification passed successfully

## How to run
1. Install dependencies:
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   ```
2. Run the verification script:
   ```bash
   python3 verify_project.py
   ```
3. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Notes
- The project is configured to run in the base Python environment; `.venv` is optional.
- For LLM report generation, install Ollama and pull a supported model such as `llama3`.
- Existing documentation in `README.md` focuses on data collection, while this file captures the full project scope.


## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Setup
1. Install Python dependencies into your base environment:
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   ```
   If you prefer user installs instead of system-wide installs:
   ```bash
   python3 -m pip install --user -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your API credentials.
   - If you use Hugging Face instead of local Ollama, set `HUGGINGFACE_API_KEY` in `.env`.
3. Run the sample ingestion script:
   ```bash
   python run_data_collection.py --statsbomb-competition 2 --statsbomb-season 27 --max-matches 5
   ```
   This downloads the first 5 matches for the Premier League open dataset.

   To download specific match IDs instead:
   ```bash
   python run_data_collection.py --statsbomb-match-ids 3754217,3754218
   ```

4. Start the Streamlit UI:
   ```bash
   streamlit run streamlit_app.py
   ```

## Testing

Use the smoke test to confirm the full pipeline works before deploying or adding new data:
```bash
python verify_project.py
```

If you want to run the app locally with Hugging Face instead of Ollama, set:
```bash
export HUGGINGFACE_API_KEY="your_huggingface_api_key"
streamlit run streamlit_app.py
```

## Streamlit Cloud deployment

This app can be deployed on Streamlit Community Cloud.

1. Push your repository to GitHub.
2. Open https://share.streamlit.io and sign in with GitHub.
3. Create a new app and select this repository.
4. Set the repository branch to `main` (or `master`) and the main file to `streamlit_app.py`.
5. In the Streamlit app settings, add a secret for:
   - `HUGGINGFACE_API_KEY`

If you want the LLM report feature in Streamlit Cloud, use your Hugging Face API key there. The app reads it via `os.getenv('HUGGINGFACE_API_KEY')`.

If you prefer a contained environment locally:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer a contained environment, create one first:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Note: This project is configured to run in your base Python environment, but using a virtual environment is recommended for isolation. The `.venv` folder is not required for execution and can be ignored or removed if you do not want to use a virtual environment.
>
> A polished end-to-end project summary is available in `PROJECT_SUMMARY.md`.

The UI allows you to select a season and team, then view deeper performance analytics from the local StatsBomb data. It now includes:
- xGBoost-based xG modeling from shot events
- K-Means tactical clustering of team event signatures
- shot maps via `mplsoccer`
- optional LLM coaching report generation using Ollama and a `llama3` model

To enable the LLM report feature:
- install Ollama on your Mac
- pull a compatible model, for example:
  ```bash
  ollama pull llama3
  ```

## Notes
- StatsBomb open event data is available via `statsbombpy`.
- `football-data.org` and `API-Football` require API keys for the free tier.
- `Open Football Data` can be accessed via raw GitHub files.
