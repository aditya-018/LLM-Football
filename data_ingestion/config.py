import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class DataConfig:
    BASE_DIR = Path(os.getenv('BASE_DIR', Path.cwd()))
    DATA_DIR = Path(os.getenv('DATA_DIR', BASE_DIR / 'data')).expanduser().resolve()

    STATSBOMB_DATA_DIR = Path(os.getenv('STATSBOMB_DATA_DIR', DATA_DIR / 'statsbomb')).expanduser().resolve()
    FOOTBALL_DATA_ORG_API_KEY = os.getenv('FOOTBALL_DATA_ORG_API_KEY', '')

    API_FOOTBALL_API_KEY = os.getenv('API_FOOTBALL_API_KEY', '')
    API_FOOTBALL_API_HOST = os.getenv('API_FOOTBALL_API_HOST', 'v3.football.api-sports.io')

    OPEN_FOOTBALL_DATA_BASE_URL = os.getenv(
        'OPEN_FOOTBALL_DATA_BASE_URL',
        'https://raw.githubusercontent.com/openfootball/football.json/master'
    )

    @classmethod
    def ensure_directories(cls):
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.STATSBOMB_DATA_DIR.mkdir(parents=True, exist_ok=True)

