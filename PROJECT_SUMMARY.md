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
