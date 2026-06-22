# Football Analytics Data Collection

This project implements the initial data collection layer for a football analytics and coaching report platform.

## Goals
- Collect open football event and match data from multiple sources
- Provide a reusable ingestion pipeline for StatsBomb, football-data.org, API-Football, and Open Football Data
- Store data locally for downstream analysis and report generation

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
