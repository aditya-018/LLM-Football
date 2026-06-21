import warnings
from pathlib import Path

import pandas as pd

from analytics.xgboost_model import (
    collect_shot_events,
    load_xg_model,
    summarize_match_xg,
    train_xg_model,
)
from analytics.tactical_clustering import build_team_profiles, cluster_teams
from llm.ollama_report import build_coaching_report_prompt, has_ollama


def main():
    warnings.filterwarnings('ignore')
    data_dir = Path('data/statsbomb')
    model_path = Path('data/models/xg_model.joblib')

    print('1) Checking StatsBomb event files...')
    shot_files = list(data_dir.glob('statsbomb_match_*_events.json'))
    print(f'   event files found: {len(shot_files)}')

    print('2) Collecting shot events...')
    shots = collect_shot_events(data_dir)
    print(f'   shot records: {len(shots)}')

    print('3) Training xG model...')
    model = train_xg_model(data_dir, save_path=model_path)
    print(f'   model metrics: {model.metrics_}')

    print('4) Loading xG model...')
    loaded_model = load_xg_model(model_path)
    print(f'   loaded model type: {type(loaded_model).__name__}')

    if len(shot_files) > 0:
        print('5) Summarizing xG for a sample match...')
        sample_path = shot_files[0]
        events = pd.read_json(sample_path, orient='records')
        summary = summarize_match_xg(events, loaded_model)
        print(f'   summary keys: {list(summary.keys())}')
        print(f'   team_xg keys: {list(summary["team_xg"].keys())}')
        print(f'   shot rows: {len(summary["shot_data"])}')

    print('6) Building tactical team profiles...')
    profiles = build_team_profiles(data_dir)
    print(f'   profile shape: {profiles.shape}')

    print('7) Clustering team profiles...')
    clustered = cluster_teams(profiles)
    print(f'   clustered shape: {clustered.shape}')
    print(f'   cluster labels: {sorted(clustered["cluster"].unique().tolist())}')

    print('8) Generating LLM prompt preview...')
    prompt = build_coaching_report_prompt(
        team='Arsenal',
        opponent='Chelsea',
        season_name='2024/25',
        summary={'matches': 5, 'wins': 3, 'draws': 1, 'losses': 1, 'goals_for': 10, 'goals_against': 5, 'pass_completion': 0.82, 'shots': 58},
        xg_summary={'team_xg': {'Arsenal': 6.2, 'Chelsea': 4.5}, 'total_xg': 10.7},
        cluster_label=1,
        cluster_desc={1: {'evt_pass': 0.30, 'evt_shot': 0.12}},
    )
    print(f'   prompt length: {len(prompt)}')
    print(f'   Ollama available: {has_ollama()}')

    print('\nAll smoke tests completed successfully.')


if __name__ == '__main__':
    main()
