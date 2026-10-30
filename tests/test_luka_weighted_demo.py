import unittest
from scripts.luka_weighted_demo import build_prior_team_games
from tests.test_opponent_context import logs


class LukaDemoTests(unittest.TestCase):
    def test_full_season_prior_games_include_march_after_development_cutoff(self):
        teams=build_prior_team_games(logs('2026-03-01'),None,'2026-03-02')
        self.assertEqual(len(teams),2)
        self.assertEqual(next(r for r in teams if r['team_id']=='AAA')['points_for'],80)

    def test_target_and_future_outcomes_are_not_accessed(self):
        class Hidden(dict):
            def __getitem__(self,key):
                if key in ['PTS','MIN']: raise AssertionError('Current/future outcome accessed')
                return super().__getitem__(key)
        before=logs('2026-03-01')
        hidden=[Hidden(r) for day in ['2026-03-02','2026-03-03'] for r in logs(day)]
        self.assertEqual(build_prior_team_games(before+hidden,None,'2026-03-02'),build_prior_team_games(before,None,'2026-03-02'))

    def test_missing_side_duplicate_and_nonfinite_are_rejected(self):
        rows=logs('2026-03-01')
        for bad in [rows[:2],rows+rows[:1],rows+[dict(rows[0],PLAYER='Other',PTS='nan')]]:
            with self.assertRaises(ValueError): build_prior_team_games(bad,None,'2026-03-02')

    def test_roster_coverage_is_checked_and_previous_season_excluded(self):
        rows=logs('2026-03-01')
        self.assertEqual(build_prior_team_games(logs('2025-03-01'),None,'2026-03-02'),[])
        with self.assertRaises(ValueError): build_prior_team_games(rows,{},'2026-03-02')
