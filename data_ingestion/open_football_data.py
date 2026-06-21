import json
from pathlib import Path
from typing import Any
import requests

class OpenFootballDataLoader:
    def __init__(self, base_url: str, save_dir: str | Path):
        self.base_url = base_url.rstrip('/')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def download_competition_file(self, remote_path: str, filename: str | None = None):
        """Download a competition JSON file from the Open Football Data GitHub repository."""
        url = f"{self.base_url}/{remote_path.lstrip('/')}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        filename = filename or Path(remote_path).name
        path = self.save_dir / filename
        with path.open('w', encoding='utf-8') as handle:
            json.dump(response.json(), handle, indent=2)
        return path
