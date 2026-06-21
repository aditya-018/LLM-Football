from __future__ import annotations

from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier

SHOT_FEATURES = [
    'location_x',
    'location_y',
    'distance',
    'angle',
    'shot_body_part',
    'shot_type',
    'shot_technique',
    'shot_one_on_one',
    'under_pressure',
]


def _normalize_value(value):
    if isinstance(value, str):
        return value.strip()
    return value


def _extract_location(location):
    if isinstance(location, (list, tuple)) and len(location) >= 2:
        return float(location[0]), float(location[1])
    return np.nan, np.nan


def _compute_distance_angle(x: float, y: float) -> tuple[float, float]:
    if math.isnan(x) or math.isnan(y):
        return np.nan, np.nan
    dx = 120.0 - x
    dy = abs(40.0 - y)
    distance = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) if dx != 0 else math.pi / 2
    return distance, angle


def _prepare_shot_frame(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events['type_name'] = events['type'].astype(str)
    shots = events[events['type_name'] == 'Shot'].copy()
    if shots.empty:
        return shots

    shots[['location_x', 'location_y']] = pd.DataFrame(
        shots['location'].apply(lambda loc: _extract_location(loc)).tolist(), index=shots.index
    )
    shots[['distance', 'angle']] = pd.DataFrame(
        shots.apply(lambda row: _compute_distance_angle(row['location_x'], row['location_y']), axis=1).tolist(),
        index=shots.index,
    )
    for col in ['shot_body_part', 'shot_type', 'shot_technique']:
        if col in shots.columns:
            shots[col] = shots[col].apply(_normalize_value).astype('object')
        else:
            shots[col] = None

    if 'shot_one_on_one' in shots.columns:
        shots['shot_one_on_one'] = shots['shot_one_on_one'].fillna(False).astype(bool)
    else:
        shots['shot_one_on_one'] = False

    if 'under_pressure' in shots.columns:
        shots['under_pressure'] = shots['under_pressure'].fillna(False).astype(bool)
    else:
        shots['under_pressure'] = False

    shots['goal'] = shots['shot_outcome'].astype(str).str.lower().eq('goal')
    shots = shots.dropna(subset=['location_x', 'location_y', 'distance', 'angle'])
    return shots


def _prepare_feature_matrix(shots: pd.DataFrame) -> pd.DataFrame:
    if shots.empty:
        return shots
    feature_matrix = shots[SHOT_FEATURES].copy()
    feature_matrix['shot_body_part'] = feature_matrix['shot_body_part'].fillna('unknown')
    feature_matrix['shot_type'] = feature_matrix['shot_type'].fillna('unknown')
    feature_matrix['shot_technique'] = feature_matrix['shot_technique'].fillna('unknown')
    return feature_matrix


def collect_shot_events(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(data_dir).glob('statsbomb_match_*_events.json')):
        df = pd.read_json(path, orient='records')
        shots = _prepare_shot_frame(df)
        if not shots.empty:
            shots['source_match_id'] = int(path.stem.split('_')[2])
            frames.append(shots)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def train_xg_model(data_dir: Path, save_path: Path | None = None, test_size: float = 0.15, random_state: int = 42) -> Pipeline:
    shots = collect_shot_events(data_dir)
    if shots.empty:
        raise ValueError('No shot events found for xG modeling. Collect StatsBomb match event files first.')

    X = _prepare_feature_matrix(shots)
    y = shots['goal'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    numeric_features = ['location_x', 'location_y', 'distance', 'angle']
    categorical_features = ['shot_body_part', 'shot_type', 'shot_technique', 'shot_one_on_one', 'under_pressure']

    try:
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', one_hot_encoder, categorical_features),
        ],
        remainder='drop',
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_estimators=100, max_depth=4)),
    ])

    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    try:
        auc = float(roc_auc_score(y_test, y_pred_proba))
        loss = float(log_loss(y_test, y_pred_proba))
    except Exception:
        auc, loss = None, None

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)

    metrics = {
        'training_samples': int(len(X_train)),
        'validation_samples': int(len(X_test)),
        'roc_auc': auc,
        'log_loss': loss,
    }
    model.metrics_ = metrics
    return model


def load_xg_model(model_path: Path) -> Pipeline:
    if not model_path.exists():
        raise FileNotFoundError(f'xG model file not found: {model_path}')
    return joblib.load(model_path)


def predict_shot_xg(model: Pipeline, shots: pd.DataFrame) -> pd.Series:
    features = _prepare_feature_matrix(shots)
    if features.empty:
        return pd.Series([], dtype=float)
    return pd.Series(model.predict_proba(features)[:, 1], index=features.index)


def summarize_match_xg(events: pd.DataFrame, model: Pipeline) -> dict:
    shots = _prepare_shot_frame(events)
    if shots.empty:
        return {'total_shots': 0, 'total_xg': 0.0, 'team_xg': {}, 'shot_data': pd.DataFrame()}

    shot_xg = predict_shot_xg(model, shots)
    shots = shots.loc[shot_xg.index].copy()
    shots['xg_model'] = shot_xg
    team_xg = shots.groupby('team')['xg_model'].sum().to_dict()
    return {
        'total_shots': int(len(shots)),
        'total_xg': float(shots['xg_model'].sum()),
        'team_xg': {team: float(xg) for team, xg in team_xg.items()},
        'shot_data': shots,
    }
