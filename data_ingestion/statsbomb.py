from pathlib import Path
import importlib
import pandas as pd
import json

# statsbombpy exposes functional API under the `sb` submodule (not via SB class)
try:
    from statsbombpy import sb as sb
except Exception:
    sb = importlib.import_module('statsbombpy.sb')


class StatsBombDataLoader:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def competitions(self):
        """List available competitions in the StatsBomb open dataset."""
        return sb.competitions()

    def matches(self, competition_id: int, season_id: int):
        """Retrieve match metadata for a competition and season."""
        return sb.matches(competition_id=competition_id, season_id=season_id)

    def save_match_list(self, competition_id: int, season_id: int, save_path: str | Path | None = None) -> Path:
        """Download and save match list metadata for a competition/season."""
        matches_df = self.matches(competition_id=competition_id, season_id=season_id)
        save_path = Path(save_path or self.data_dir / f"statsbomb_matches_{competition_id}_{season_id}.json")
        matches_df.to_json(save_path, orient='records', indent=2, force_ascii=False)
        return save_path

    def match_events(self, match_id: int) -> pd.DataFrame:
        """Retrieve event data for a specific match as a DataFrame."""
        return sb.events(match_id=match_id)

    def save_match_metadata(self, match_id: int, rows: pd.DataFrame | None = None, save_path: str | Path | None = None) -> Path:
        """Save metadata for a specific match identified by match_id."""
        if rows is None:
            raise ValueError('Rows DataFrame is required to save match metadata')
        match_row = rows[rows['match_id'] == match_id]
        if match_row.empty:
            raise ValueError(f'Match {match_id} not found in provided rows')
        row_dict = match_row.iloc[0].to_dict()
        save_path = Path(save_path or self.data_dir / f"statsbomb_match_{match_id}_metadata.json")
        with save_path.open('w', encoding='utf-8') as fh:
            json.dump(row_dict, fh, indent=2, default=str)
        return save_path

    def save_match_events(self, match_id: int, save_path: str | Path | None = None) -> Path:
        """Download and persist event data for a match to JSON and return the path."""
        events_df = self.match_events(match_id)
        save_path = Path(save_path or self.data_dir / f"statsbomb_match_{match_id}_events.json")
        # pandas to_json may not preserve nested dicts cleanly; convert to records via orient
        events_df.to_json(save_path, orient='records', indent=2, force_ascii=False)
        return save_path

    def _load_events_from_path(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        return pd.read_json(path, orient='records')

    def validate_events(self, events: pd.DataFrame | str | Path) -> dict:
        """Validate a match events DataFrame or JSON file and return a summary dict.

        Checks performed:
        - row count
        - unique teams and players
        - presence and basic ranges of location coordinates
        - top event types
        """
        if not isinstance(events, pd.DataFrame):
            events = self._load_events_from_path(events)

        summary: dict = {}
        summary['rows'] = int(len(events))

        # team and player counts
        if 'team' in events.columns:
            try:
                summary['unique_teams'] = int(events['team'].nunique())
            except Exception:
                summary['unique_teams'] = None
        else:
            summary['unique_teams'] = None

        if 'player' in events.columns:
            try:
                summary['unique_players'] = int(events['player'].nunique())
            except Exception:
                summary['unique_players'] = None
        else:
            summary['unique_players'] = None

        # location handling: either 'location' list or x/y columns
        x_vals = None
        y_vals = None
        if 'location' in events.columns:
            # location is often a list [x,y]
            locs = events['location'].apply(lambda v: (v[0], v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else (None, None))
            x_vals = locs.map(lambda t: t[0])
            y_vals = locs.map(lambda t: t[1])
        else:
            for cand_x in ['x', 'start_x', 'end_x']:
                if cand_x in events.columns:
                    x_vals = events[cand_x]
                    break
            for cand_y in ['y', 'start_y', 'end_y']:
                if cand_y in events.columns:
                    y_vals = events[cand_y]
                    break

        if x_vals is not None and y_vals is not None:
            try:
                x_num = pd.to_numeric(x_vals, errors='coerce')
                y_num = pd.to_numeric(y_vals, errors='coerce')
                summary['location_missing'] = int(x_num.isna().sum() + y_num.isna().sum())
                if x_num.notna().any():
                    summary['x_range'] = (float(x_num.min()), float(x_num.max()))
                if y_num.notna().any():
                    summary['y_range'] = (float(y_num.min()), float(y_num.max()))
            except Exception:
                summary['location_missing'] = None
                summary['x_range'] = None
                summary['y_range'] = None
        else:
            summary['location_missing'] = None
            summary['x_range'] = None
            summary['y_range'] = None

        # event type distribution
        if 'type' in events.columns:
            def _type_name(v):
                if isinstance(v, dict):
                    return v.get('name') or str(v)
                return v

            try:
                type_series = events['type'].apply(_type_name).astype('object')
                summary['top_event_types'] = type_series.value_counts().head(10).to_dict()
            except Exception:
                summary['top_event_types'] = None
        else:
            summary['top_event_types'] = None

        # sample events
        try:
            sample = events.head(5).to_dict(orient='records')
            summary['sample_events'] = sample
        except Exception:
            summary['sample_events'] = None

        return summary

    def validate_match_events(self, match_id: int, save_summary: bool = True) -> dict:
        """Load, validate and optionally save a short JSON summary for a given match id."""
        df = self.match_events(match_id)
        summary = self.validate_events(df)
        if save_summary:
            out_path = self.data_dir / f'statsbomb_match_{match_id}_summary.json'
            with out_path.open('w', encoding='utf-8') as fh:
                json.dump(summary, fh, indent=2)
            summary['_summary_path'] = str(out_path)
        return summary

    def preview_match_events(self, match_id: int, n: int = 10) -> pd.DataFrame:
        """Return the first `n` event rows for quick previewing."""
        df = self.match_events(match_id)
        return df.head(n)
