import json
from pathlib import Path
from typing import Any
import requests

class APIFootballClient:
    BASE_URL = 'https://v3.football.api-sports.io'

    def __init__(self, api_key: str, api_host: str, save_dir: str | Path):
        if not api_key:
            raise ValueError('API-Football API key is required')
        self.api_key = api_key
        self.api_host = api_host
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _request(self, endpoint: str, params: dict[str, Any] | None = None):
        url = f"{self.BASE_URL}{endpoint}"
        headers = {
            'x-apisports-key': self.api_key,
            'x-apisports-host': self.api_host,
            'User-Agent': 'football-analytics-data-collector/1.0'
        }
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def _save_json(self, data: dict[str, Any], filename: str):
        path = self.save_dir / filename
        with path.open('w', encoding='utf-8') as handle:
            json.dump(data, handle, indent=2)
        return path

    def get_fixtures(self, league_id: int, season: int, status: str = 'NS'):
        params = {'league': league_id, 'season': season, 'status': status}
        data = self._request('/fixtures', params)
        return self._save_json(data, f'api_football_fixtures_{league_id}_{season}.json')

    def get_team_players(self, team_id: int, season: int):
        params = {'team': team_id, 'season': season}
        data = self._request('/players', params)
        return self._save_json(data, f'api_football_team_{team_id}_players_{season}.json')
