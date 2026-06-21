from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_EVENT_TYPES = [
    'Pass',
    'Carry',
    'Dribble',
    'Shot',
    'Pressure',
    'Ball Recovery',
    'Duel',
    'Clearance',
    'Block',
    'Interception',
    'Foul Committed',
    'Goal Keeper',
]


def _type_name(value):
    if isinstance(value, dict):
        return value.get('name') or str(value)
    return str(value)


def collect_team_event_signatures(data_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(data_dir).glob('statsbomb_match_*_events.json')):
        df = pd.read_json(path, orient='records')
        if df.empty:
            continue
        df['type_name'] = df['type'].apply(_type_name).astype(str)
        for team, team_events in df.groupby('team'):
            counts = team_events['type_name'].value_counts()
            row = {'team': team, 'match_id': int(path.stem.split('_')[2]), 'total_events': int(len(team_events))}
            for event_type in DEFAULT_EVENT_TYPES:
                row[f'evt_{event_type}'] = int(counts.get(event_type, 0))
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    profiles = pd.DataFrame(rows)
    profiles = profiles.set_index(['team', 'match_id'])
    return profiles


def build_team_profiles(data_dir: Path) -> pd.DataFrame:
    profiles = collect_team_event_signatures(data_dir)
    if profiles.empty:
        return profiles

    normalized = profiles.div(profiles['total_events'], axis=0).fillna(0)
    return normalized.drop(columns=['total_events'])


def cluster_teams(team_profiles: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    if team_profiles.empty:
        return team_profiles

    feature_cols = [c for c in team_profiles.columns if c.startswith('evt_')]
    if not feature_cols:
        return team_profiles

    averaged = team_profiles.groupby('team')[feature_cols].mean()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(averaged)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(scaled)
    averaged['cluster'] = labels
    averaged['cluster'] = averaged['cluster'].astype(int)
    return averaged


def describe_cluster_centers(clustered_profiles: pd.DataFrame) -> dict:
    if clustered_profiles.empty or 'cluster' not in clustered_profiles.columns:
        return {}

    cluster_summaries = {}
    feature_cols = [c for c in clustered_profiles.columns if c.startswith('evt_')]
    for cluster_id, group in clustered_profiles.groupby('cluster'):
        center = group[feature_cols].mean().sort_values(ascending=False).head(5)
        cluster_summaries[int(cluster_id)] = center.to_dict()
    return cluster_summaries
