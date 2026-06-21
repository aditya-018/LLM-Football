import argparse
from data_ingestion.config import DataConfig
from data_ingestion.pipeline import DataCollectionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Run football data collection pipeline.')
    parser.add_argument('--statsbomb-competition', type=int, default=2, help='StatsBomb competition ID')
    parser.add_argument('--statsbomb-season', type=int, default=27, help='StatsBomb season ID')
    parser.add_argument('--statsbomb-match-ids', type=str, default='', help='Comma-separated StatsBomb match IDs to download explicitly')
    parser.add_argument('--max-matches', type=int, default=5, help='Maximum matches to download from StatsBomb')
    parser.add_argument('--football-data-org-competition', type=str, default='PL', help='football-data.org competition code')
    parser.add_argument('--api-football-league', type=int, default=39, help='API-Football league ID')
    parser.add_argument('--api-football-season', type=int, default=2024, help='API-Football season')
    parser.add_argument('--open-football-path', type=str, default='england/2024-25/england.json', help='Open Football Data remote path')
    return parser.parse_args()


def main():
    args = parse_args()
    DataConfig.ensure_directories()
    pipeline = DataCollectionPipeline(DataConfig)

    print('Collecting StatsBomb match events...')
    match_ids = [int(x) for x in args.statsbomb_match_ids.split(',') if x.strip()] if args.statsbomb_match_ids else None
    sb_result = pipeline.collect_statsbomb_match_events(
        competition_id=args.statsbomb_competition,
        season_id=args.statsbomb_season,
        max_matches=args.max_matches,
        match_ids=match_ids,
    )
    print('  Match list metadata saved at:', sb_result['match_list_path'])
    for match_info in sb_result['matches']:
        print('  Saved events:', match_info['events_path'])
        if match_info.get('metadata_path'):
            print('    Saved metadata:', match_info['metadata_path'])
        try:
            summary = pipeline.statsbomb.validate_match_events(int(match_info['match_id']))
            print('    Validation summary saved at:', summary.get('_summary_path'))
            short_keys = {k: v for k, v in summary.items() if k not in ['sample_events', '_summary_path']}
            print('    Summary:', short_keys)
        except Exception as e:
            print('    Validation failed for match', match_info['match_id'], '->', e)

    # football-data.org (optional)
    if pipeline.football_data_org is not None:
        try:
            print('Collecting football-data.org competition data...')
            fd_paths = pipeline.collect_football_data_org_competition(
                competition_id=args.football_data_org_competition
            )
            print('  Saved:', fd_paths)
        except Exception as e:
            print('  football-data.org collection failed:', e)
    else:
        print('Skipping football-data.org collection (no API key configured).')

    # API-Football (optional)
    if pipeline.api_football is not None:
        try:
            print('Collecting API-Football fixtures...')
            api_paths = pipeline.collect_api_football_fixture_data(
                league_id=args.api_football_league,
                season=args.api_football_season,
            )
            print('  Saved:', api_paths)
        except Exception as e:
            print('  API-Football collection failed:', e)
    else:
        print('Skipping API-Football collection (no API key configured).')

    print('Collecting Open Football Data file...')
    try:
        open_path = pipeline.collect_open_football_competition(args.open_football_path)
        print('  Saved:', open_path)
    except Exception as e:
        print('  Open Football Data download failed:', e)

    print('Data collection complete.')
    print('Directory summary:', pipeline.summary())


if __name__ == '__main__':
    main()
