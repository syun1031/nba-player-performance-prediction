from datetime import datetime,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
from scripts.update_current_season import update,collection_day


class DailyUpdateTests(unittest.TestCase):
    def test_uses_eastern_date_not_taiwan_date(self):
        self.assertEqual(collection_day(datetime(2026,11,5,0,10,tzinfo=timezone.utc)),'2026-11-04')
        self.assertEqual(collection_day(datetime(2026,11,5,9,tzinfo=timezone.utc)),'2026-11-05')

    def test_failed_collection_never_builds_stale_success(self):
        with TemporaryDirectory() as folder:
            with patch('scripts.update_current_season.subprocess.run',side_effect=RuntimeError('NBA unavailable')),patch('scripts.update_current_season.build_future') as build:
                with self.assertRaises(RuntimeError):update(Path(folder),now=datetime(2026,11,4,9,tzinfo=timezone.utc))
                build.assert_not_called()

    def test_success_exit_with_unchanged_pointer_is_rejected(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);out=root/'outputs/nba_2026_27';out.mkdir(parents=True)
            (out/'latest_success.json').write_text(json.dumps({'snapshot':'stale'}))
            with patch('scripts.update_current_season.subprocess.run'),patch('scripts.update_current_season.build_future') as build:
                with self.assertRaisesRegex(ValueError,'fresh'):update(root,now=datetime(2026,11,4,9,tzinfo=timezone.utc))
                build.assert_not_called()

    def test_new_snapshot_cannot_drop_an_old_completed_game(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);snap=root/'outputs/nba_2026_27/fresh';snap.mkdir(parents=True)
            old=root/'outputs/future_prediction';old.mkdir(parents=True)
            (old/'manifest.json').write_text(json.dumps({'completed_games':1}))
            (old/'team_games.csv').write_text('nba_game_id\n0022600001\n')
            (snap/'manifest.json').write_text(json.dumps({'complete':True,'errors':[],'through':'2026-11-04','completed_games_collected':1}))
            (snap/'team_games.csv').write_text('nba_game_id\n0022600002\n')
            def collected(*a,**k):(snap.parent/'latest_success.json').write_text(json.dumps({'snapshot':'fresh'}))
            with patch('scripts.update_current_season.subprocess.run',side_effect=collected),patch('scripts.update_current_season.build_future') as build:
                with self.assertRaisesRegex(ValueError,'disappeared'):update(root,now=datetime(2026,11,4,9,tzinfo=timezone.utc))
                build.assert_not_called()
