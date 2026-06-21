from pathlib import Path
from typing import List

from .config import DataConfig
from .statsbomb import StatsBombDataLoader
from .football_data_org import FootballDataOrgClient
from .api_football import APIFootballClient
from .open_football_data import OpenFootballDataLoader

class DataCollectionPipeline:
    def __init__(self, config: DataConfig):
        self.config = config
        self.statsbomb = StatsBombDataLoader(config.STATSBOMB_DATA_DIR)
        # Instantiate optional API clients only if API keys are provided
        if config.FOOTBALL_DATA_ORG_API_KEY:
            self.football_data_org = FootballDataOrgClient(
                api_key=config.FOOTBALL_DATA_ORG_API_KEY,
                save_dir=config.DATA_DIR / 'football_data_org'
            )
        else:
            self.football_data_org = None

        if config.API_FOOTBALL_API_KEY:
            self.api_football = APIFootballClient(
                api_key=config.API_FOOTBALL_API_KEY,
                api_host=config.API_FOOTBALL_API_HOST,
                save_dir=config.DATA_DIR / 'api_football'
            )
        else:
            self.api_football = None
        self.open_football = OpenFootballDataLoader(
            base_url=config.OPEN_FOOTBALL_DATA_BASE_URL,
            save_dir=config.DATA_DIR / 'open_football'
        )

    def collect_statsbomb_match_events(self, competition_id: int, season_id: int, max_matches: int = 5, match_ids: list[int] | None = None):
        matches = self.statsbomb.matches(competition_id=competition_id, season_id=season_id)
        match_list_path = self.statsbomb.save_match_list(competition_id, season_id)
        match_ids_to_download: list[int]
        if match_ids:
            match_ids_to_download = [int(mid) for mid in match_ids]
            matches = matches[matches['match_id'].isin(match_ids_to_download)]
        else:
            selected = matches.head(max_matches)
            match_ids_to_download = [int(mid) for mid in selected['match_id']]
            matches = selected

        saved = []
        for match_id in match_ids_to_download:
            events_path = self.statsbomb.save_match_events(match_id)
            try:
                match_metadata_path = self.statsbomb.save_match_metadata(match_id, rows=matches)
            except Exception:
                match_metadata_path = None
            saved.append({
                'match_id': int(match_id),
                'events_path': str(events_path),
                'metadata_path': str(match_metadata_path) if match_metadata_path else None,
            })
        return {
            'match_list_path': str(match_list_path),
            'matches': saved,
        }

    def collect_football_data_org_competition(self, competition_id: str, date_from: str | None = None, date_to: str | None = None):
        if not self.football_data_org:
            raise ValueError('football-data.org client not configured. Set FOOTBALL_DATA_ORG_API_KEY in .env to enable this.')
        matches_path = self.football_data_org.get_matches(competition_id, date_from, date_to)
        standings_path = self.football_data_org.get_standings(competition_id)
        return {'matches': matches_path, 'standings': standings_path}

    def collect_api_football_fixture_data(self, league_id: int, season: int, status: str = 'NS'):
        if not self.api_football:
            raise ValueError('API-Football client not configured. Set API_FOOTBALL_API_KEY in .env to enable this.')
        return self.api_football.get_fixtures(league_id, season, status)

    def collect_open_football_competition(self, remote_path: str):
        return self.open_football.download_competition_file(remote_path)

    def summary(self):
        return {
            'statsbomb_dir': str(self.config.STATSBOMB_DATA_DIR),
            'football_data_org_dir': str(self.config.DATA_DIR / 'football_data_org'),
            'api_football_dir': str(self.config.DATA_DIR / 'api_football'),
            'open_football_dir': str(self.config.DATA_DIR / 'open_football'),
        }
