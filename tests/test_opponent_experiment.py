import unittest
from scripts.experiment_opponent_context import augment_opponent_rows
from scripts.opponent_context import build_team_games
from tests.test_opponent_context import logs


class OpponentExperimentTests(unittest.TestCase):
    def test_common_cohort_exclusion_is_only_five_game_gate_and_preserves_values(self):
        teams=build_team_games(sum([logs(f'2025-10-{d:02d}') for d in range(21,27)],[]))
        rows=[dict(player_id='P',game_id='AAA @ BBB|'+day,GAMEDATE=day,split='train',target_pts=point,
                   pts_mean_5='12.34') for day,point in [('2025-10-25',99),('2025-10-26',0)]]
        good,excluded=augment_opponent_rows(rows,teams)
        self.assertEqual(len(good),1); self.assertEqual(good[0]['target_pts'],0)
        self.assertEqual(good[0]['pts_mean_5'],'12.34')
        self.assertEqual(excluded[0]['target_pts'],99)
        self.assertEqual(excluded[0]['opponent_history_count'],4)
        self.assertEqual(excluded[0]['opponent_exclusion_reason'],'insufficient_opponent_history')

    def test_no_test_dates_or_duplicate_samples_are_accepted(self):
        teams=build_team_games(logs())
        row=dict(player_id='P',game_id='AAA @ BBB|2025-10-25',GAMEDATE='2025-10-25',split='train')
        with self.assertRaises(ValueError): augment_opponent_rows([row,row],teams)
        with self.assertRaises(ValueError): augment_opponent_rows([dict(row,GAMEDATE='2026-03-01',split='test')],teams)


if __name__=='__main__':
    unittest.main()
