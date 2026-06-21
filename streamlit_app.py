import json
import logging
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from mplsoccer import Pitch

from analytics.xgboost_model import load_xg_model, train_xg_model, summarize_match_xg
from analytics.tactical_clustering import build_team_profiles, cluster_teams, describe_cluster_centers
from llm.ollama_report import build_coaching_report_prompt, generate_ollama_report, has_ollama, has_huggingface

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path('data/statsbomb')
MODEL_DIR = Path('data/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
XG_MODEL_PATH = MODEL_DIR / 'xg_model.joblib'


def load_match_list_files():
    files = sorted(DATA_DIR.glob('statsbomb_matches_*.json'))
    match_lists = []
    for file in files:
        try:
            df = pd.read_json(file, orient='records')
            if df.empty:
                continue
            season_name = str(df.loc[0, 'season']) if 'season' in df.columns else 'unknown'
            competition_name = str(df.loc[0, 'competition_name']) if 'competition_name' in df.columns else 'unknown'
            competition_id = int(df.loc[0, 'competition_id']) if 'competition_id' in df.columns else None
            season_id = int(df.loc[0, 'season_id']) if 'season_id' in df.columns else None
            match_lists.append({
                'path': file,
                'competition_name': competition_name,
                'season_name': season_name,
                'competition_id': competition_id,
                'season_id': season_id,
                'match_count': len(df),
            })
        except Exception:
            continue
    return match_lists


def season_year(season_name: str) -> int | None:
    if not season_name:
        return None
    if '/' in season_name:
        try:
            return int(season_name.split('/')[0])
        except ValueError:
            return None
    try:
        return int(season_name)
    except ValueError:
        return None


def load_match_list(path: Path) -> pd.DataFrame:
    return pd.read_json(path, orient='records')


def load_match_events(match_id: int) -> pd.DataFrame:
    path = DATA_DIR / f'statsbomb_match_{match_id}_events.json'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, orient='records')


def load_team_events(match_ids: list[int], team_name: str) -> pd.DataFrame:
    frames = []
    for match_id in match_ids:
        df = load_match_events(match_id)
        if df.empty:
            continue
        if 'team' in df.columns:
            df = df[df['team'] == team_name]
        else:
            continue
        df['match_id'] = match_id
        frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def load_match_events_for_matches(match_ids: list[int]) -> pd.DataFrame:
    frames = []
    for match_id in match_ids:
        df = load_match_events(match_id)
        if df.empty:
            continue
        df['match_id'] = match_id
        frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def compute_team_results(team: str, df: pd.DataFrame) -> pd.DataFrame:
    def _result(row):
        home = row['home_team'] == team
        if home:
            gf = row['home_score']
            ga = row['away_score']
        else:
            gf = row['away_score']
            ga = row['home_score']
        if gf > ga:
            return 'Win'
        if gf == ga:
            return 'Draw'
        return 'Loss'

    results = df.copy()
    results['team'] = team
    results['opponent'] = results.apply(lambda r: r['away_team'] if r['home_team'] == team else r['home_team'], axis=1)
    results['venue'] = results.apply(lambda r: 'Home' if r['home_team'] == team else 'Away', axis=1)
    results['goals_for'] = results.apply(lambda r: r['home_score'] if r['home_team'] == team else r['away_score'], axis=1)
    results['goals_against'] = results.apply(lambda r: r['away_score'] if r['home_team'] == team else r['home_score'], axis=1)
    results['result'] = results.apply(_result, axis=1)
    return results


def summarize_team_events(team_events: pd.DataFrame) -> dict:
    summary = {}
    if team_events.empty:
        return summary

    summary['total_events'] = len(team_events)
    summary['event_type_counts'] = team_events['type'].astype(str).value_counts().head(15)

    passes = team_events[team_events['type'] == 'Pass']
    summary['passes'] = len(passes)
    if not passes.empty and 'pass_outcome' in passes.columns:
        completed = passes['pass_outcome'].astype(str).str.lower().eq('complete')
        summary['pass_completion'] = float(completed.sum()) / len(passes) if len(passes) > 0 else 0.0
    else:
        summary['pass_completion'] = None

    shots = team_events[team_events['type'] == 'Shot']
    summary['shots'] = len(shots)
    if not shots.empty and 'shot_outcome' in shots.columns:
        on_target = shots['shot_outcome'].astype(str).str.lower().isin(['saved', 'goal', 'blocked', 'off target', 'post'])
        summary['shot_accuracy'] = float(on_target.sum()) / len(shots) if len(shots) > 0 else 0.0
    else:
        summary['shot_accuracy'] = None

    if 'player' in team_events.columns:
        summary['top_players'] = team_events['player'].astype(str).value_counts().head(10)
    else:
        summary['top_players'] = pd.Series([], dtype=int)

    if 'tactics' in team_events.columns:
        def formation(item):
            if isinstance(item, dict):
                return item.get('formation')
            if isinstance(item, str):
                try:
                    data = json.loads(item)
                    return data.get('formation')
                except Exception:
                    return None
            return None

        formations = team_events['tactics'].apply(formation).dropna().astype(str)
        summary['formations'] = formations.value_counts().head(10)
    else:
        summary['formations'] = pd.Series([], dtype=int)

    return summary


def format_metric(value):
    return f"{value:.2f}" if isinstance(value, float) else value


def plot_shot_map(shots: pd.DataFrame, title: str) -> plt.Figure:
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#a8bc95', line_color='#222222')
    fig, ax = pitch.draw(figsize=(8, 5))

    if shots.empty:
        ax.text(60, 40, 'No shot data available', ha='center', va='center', color='red', fontsize=14)
        return fig

    x = shots['location_x'].astype(float)
    y = shots['location_y'].astype(float)
    sizes = shots.get('xg_model', pd.Series(0.1, index=shots.index)).fillna(0.1).astype(float) * 200
    cmap = 'inferno'
    sc = ax.scatter(x, y, s=sizes, c=shots.get('xg_model', 0.1), cmap=cmap, edgecolors='black', alpha=0.9)
    cbar = fig.colorbar(sc, ax=ax, label='Predicted xG', shrink=0.7)
    ax.set_title(title)
    return fig


def build_team_match_summary(team_matches: pd.DataFrame, summary: dict) -> dict:
    return {
        'matches': int(len(team_matches)),
        'wins': int((team_matches['result'] == 'Win').sum()),
        'draws': int((team_matches['result'] == 'Draw').sum()),
        'losses': int((team_matches['result'] == 'Loss').sum()),
        'goals_for': int(team_matches['goals_for'].sum()),
        'goals_against': int(team_matches['goals_against'].sum()),
        'pass_completion': summary.get('pass_completion'),
        'shots': summary.get('shots'),
        'event_type_counts': summary.get('event_type_counts'),
    }


st.set_page_config(page_title='Football Team Analysis', layout='wide')
st.title('Football Team Performance & Opponent Profiling')
st.markdown(
    'This POC combines raw StatsBomb event data with xGBoost xG modeling, K-Means tactical clustering, and Ollama-powered coaching report generation.'
)

match_lists = load_match_list_files()
if not match_lists:
    st.warning('No local StatsBomb season metadata found in data/statsbomb. Collect match data first using run_data_collection.py.')
    st.info('Example: python run_data_collection.py --statsbomb-competition 2 --statsbomb-season 27 --max-matches 5')
    st.stop()

season_filters = [m for m in match_lists if season_year(m['season_name']) is None or season_year(m['season_name']) >= 2022]
if not season_filters:
    season_filters = match_lists

selected_season = st.sidebar.selectbox(
    'Available competition / season',
    options=season_filters,
    format_func=lambda x: f"{x['competition_name']} ({x['season_name']}) - {x['match_count']} matches",
)

match_list = load_match_list(Path(selected_season['path']))
teams = sorted(set(match_list['home_team'].dropna().astype(str).unique()).union(set(match_list['away_team'].dropna().astype(str).unique())))
selected_team = st.sidebar.selectbox('Select team', teams)

sidebar_model = st.sidebar.expander('Modeling & Report Tools', expanded=True)
with sidebar_model:
    model_status = 'available' if XG_MODEL_PATH.exists() else 'not trained'
    st.write('xG model status:', model_status)
    if st.button('Train xG model from local shots', key='train_xg'):
        with st.spinner('Training xG model from existing StatsBomb shot events...'):
            try:
                xg_model = train_xg_model(DATA_DIR, save_path=XG_MODEL_PATH)
                st.success('xG model trained and saved.')
                if hasattr(xg_model, 'metrics_'):
                    st.json(xg_model.metrics_)
            except Exception as exc:
                st.error(f'Failed to train xG model: {exc}')
                xg_model = None
    ollama_available = has_ollama()
    huggingface_available = has_huggingface()
    st.write('Ollama available:', 'Yes' if ollama_available else 'No')
    st.write('Hugging Face available:', 'Yes' if huggingface_available else 'No')
    if ollama_available:
        st.write('Use `ollama pull llama3` locally to enable narrative generation.')
    elif huggingface_available:
        st.write('Hugging Face is configured via HUGGINGFACE_API_KEY and will be used for report generation.')
    else:
        st.write('Install Ollama locally or set HUGGINGFACE_API_KEY to enable LLM coaching reports.')

try:
    xg_model = load_xg_model(XG_MODEL_PATH) if XG_MODEL_PATH.exists() else None
except Exception as exc:
    logger.error('Failed loading xG model: %s', exc)
    xg_model = None

team_matches = match_list[(match_list['home_team'] == selected_team) | (match_list['away_team'] == selected_team)].copy()
team_matches = compute_team_results(selected_team, team_matches)
team_matches = team_matches.sort_values(['match_date', 'match_id'], ascending=[True, True])

if team_matches.empty:
    st.warning('No matches found for this team in the selected season metadata.')
    st.stop()

# Prepare list of match IDs for reuse in multiple calls below
match_ids = team_matches['match_id'].astype(int).tolist()

team_events = load_team_events(match_ids, selected_team)
match_events = load_match_events_for_matches(match_ids)
summary = summarize_team_events(team_events)
ml_summary = build_team_match_summary(team_matches, summary)

team_profiles = build_team_profiles(DATA_DIR)
clustered_profiles = cluster_teams(team_profiles)
cluster_desc = describe_cluster_centers(clustered_profiles)
team_cluster_label = int(clustered_profiles.loc[selected_team, 'cluster']) if selected_team in clustered_profiles.index else None

xg_report = None
if xg_model is not None and not match_events.empty:
    try:
        xg_report = summarize_match_xg(match_events, xg_model)
    except Exception as exc:
        logger.warning('xG summary failed: %s', exc)
        xg_report = None

wins = int((team_matches['result'] == 'Win').sum())
draws = int((team_matches['result'] == 'Draw').sum())
losses = int((team_matches['result'] == 'Loss').sum())

with st.expander('Team summary metrics', expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.metric('Matches available', len(team_matches))
    col1.metric('Wins', wins)
    col2.metric('Draws', draws)
    col2.metric('Losses', losses)
    col3.metric('Goals for', int(team_matches['goals_for'].sum()))
    col3.metric('Goals against', int(team_matches['goals_against'].sum()))
    if summary.get('pass_completion') is not None:
        col1.metric('Pass completion', format_metric(summary.get('pass_completion')))
    if summary.get('shots') is not None:
        col2.metric('Shots', summary.get('shots'))
    if xg_report is not None:
        xg_value = xg_report['team_xg'].get(selected_team, 0.0)
        col3.metric('Est. xG for team', format_metric(xg_value))

st.markdown('### Match list')
st.dataframe(team_matches[['match_date', 'kick_off', 'venue', 'opponent', 'home_score', 'away_score', 'result', 'stadium']].rename(columns={
    'home_score': 'home', 'away_score': 'away', 'match_date': 'date', 'kick_off': 'kickoff'
}), use_container_width=True)

match_event_tabs = st.tabs(['Match Event Summary', 'Shot Map', 'Tactical Cluster', 'LLM Report'])

with match_event_tabs[0]:
    st.markdown('### Match event summary')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total team events', format_metric(summary.get('total_events', 0)))
    col2.metric('Passes', format_metric(summary.get('passes', 0)))
    col3.metric('Pass completion', format_metric(summary.get('pass_completion', 0.0)))
    col4.metric('Shots', format_metric(summary.get('shots', 0)))

    if not team_events.empty:
        st.markdown('#### Top event types')
        etype = summary.get('event_type_counts')
        df_et = None
        if isinstance(etype, dict):
            df_et = pd.DataFrame(list(etype.items()), columns=['event_type', 'count'])
        elif isinstance(etype, pd.Series):
            df_et = etype.reset_index()
            df_et.columns = ['event_type', 'count']
        if df_et is not None and not df_et.empty:
            fig = px.bar(df_et, x='event_type', y='count')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No event-type counts available for plotting.')

        st.markdown('#### Top players by event count')
        top_players = summary.get('top_players')
        df_tp = None
        if isinstance(top_players, pd.Series):
            df_tp = top_players.reset_index()
            df_tp.columns = ['player', 'count']
        elif isinstance(top_players, dict):
            df_tp = pd.DataFrame(list(top_players.items()), columns=['player', 'count'])
        if df_tp is not None and not df_tp.empty:
            fig = px.bar(df_tp, x='player', y='count')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No player event breakdown available.')

        if summary['pass_completion'] is not None:
            st.markdown('#### Pass completion over matches')
            pass_records = team_events[team_events['type'] == 'Pass'].copy()
            pass_records['completed'] = pass_records['pass_outcome'].astype(str).str.lower() == 'complete'
            if not pass_records.empty:
                pass_summary = pass_records.groupby('match_id')['completed'].agg(['sum', 'count']).reset_index()
                pass_summary['completion'] = pass_summary['sum'] / pass_summary['count']
                fig = px.bar(pass_summary, x='match_id', y='completion', labels={'match_id': 'Match ID', 'completion': 'Pass completion'})
                st.plotly_chart(fig, use_container_width=True)

        if summary.get('formations') is not None:
            formations = summary.get('formations')
            df_fm = None
            if isinstance(formations, pd.Series):
                df_fm = formations.reset_index()
                df_fm.columns = ['formation', 'count']
            elif isinstance(formations, dict):
                df_fm = pd.DataFrame(list(formations.items()), columns=['formation', 'count'])
            if df_fm is not None and not df_fm.empty:
                st.markdown('#### Formation counts from Starting XI')
                fig = px.bar(df_fm, x='formation', y='count')
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('#### Raw team match event coverage')
        st.write(f'Loaded {len(team_events)} team events from {len(match_ids)} match files.')
        st.dataframe(team_events[['match_id', 'type', 'minute', 'position', 'pass_outcome', 'shot_outcome', 'player']].head(50), use_container_width=True)
    else:
        st.warning('Team-level event data was not found for the selected matches. Ensure the corresponding event JSON files exist in data/statsbomb.')

with match_event_tabs[1]:
    st.markdown('### Shot map')
    if xg_report is not None and not xg_report['shot_data'].empty:
        fig = plot_shot_map(xg_report['shot_data'][xg_report['shot_data']['team'] == selected_team], f'{selected_team} shot map with predicted xG')
        st.pyplot(fig)
    elif not team_events.empty:
        shots = team_events[team_events['type'] == 'Shot'].copy()
        if not shots.empty:
            shots[['location_x', 'location_y']] = pd.DataFrame(shots['location'].tolist(), index=shots.index)
            fig = plot_shot_map(shots, f'{selected_team} shot map (no xG model)')
            st.pyplot(fig)
        else:
            st.info('No shot events available for this team.')
    else:
        st.info('No team shot data available to plot.')

with match_event_tabs[2]:
    st.markdown('### Tactical clustering')
    if clustered_profiles.empty:
        st.warning('Not enough event data to compute tactical clusters.')
    else:
        st.write('Team cluster assignment for the selected season and available matches:')
        st.write(f'- {selected_team} cluster: {team_cluster_label}')
        st.write('Cluster center summaries:')
        for cluster_id, cluster_stats in cluster_desc.items():
            st.markdown(f'**Cluster {cluster_id}**')
            for stat, value in cluster_stats.items():
                st.write(f'- {stat.replace("evt_", "").replace("_", " ")}: {value:.2%}')

with match_event_tabs[3]:
    st.markdown('### LLM coaching report')
    ollama_available = has_ollama()
    huggingface_available = has_huggingface()
    if not ollama_available and not huggingface_available:
        st.warning('No LLM provider is configured. Install Ollama locally or set HUGGINGFACE_API_KEY to generate coaching reports.')
        st.code('ollama pull llama3')
        st.code('export HUGGINGFACE_API_KEY="your_token_here"')
    opponent = st.selectbox('Select opponent to profile', sorted(team_matches['opponent'].unique()))
    if st.button('Generate coaching report'):
        if not ollama_available and not huggingface_available:
            st.error('Cannot generate report without Ollama or Hugging Face configured.')
        else:
            prompt = build_coaching_report_prompt(
                team=selected_team,
                opponent=opponent,
                season_name=selected_season['season_name'],
                summary=ml_summary,
                xg_summary=xg_report or {},
                cluster_label=team_cluster_label,
                cluster_desc=cluster_desc,
            )
            try:
                report = generate_ollama_report(prompt)
                st.markdown('#### Generated coaching report')
                st.write(report)
            except Exception as exc:
                st.error(f'Failed to generate coaching report: {exc}')
