import json
from pathlib import Path
from typing import Any
import requests

class FootballDataOrgClient:
    BASE_URL = 'https://api.football-data.org/v4'

    def __init__(self, api_key: str, save_dir: str | Path):
        if not api_key:
            raise ValueError('football-data.org API key is required')
        self.api_key = api_key
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _request(self, endpoint: str, params: dict[str, Any] | None = None):
        url = f"{self.BASE_URL}{endpoint}"
        headers = {
            'X-Auth-Token': self.api_key,
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

    def get_competitions(self):
        data = self._request('/competitions')
        return self._save_json(data, 'football_data_org_competitions.json')

    def get_matches(self, competition_id: str, date_from: str | None = None, date_to: str | None = None):
        params = {}
        if date_from:
            params['dateFrom'] = date_from
        if date_to:
            params['dateTo'] = date_to
        data = self._request(f'/competitions/{competition_id}/matches', params)
        return self._save_json(data, f'football_data_org_{competition_id}_matches.json')

    def get_standings(self, competition_id: str):
        data = self._request(f'/competitions/{competition_id}/standings')
        return self._save_json(data, f'football_data_org_{competition_id}_standings.json')
