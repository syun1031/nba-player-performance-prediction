import math
import unittest
from copy import deepcopy

from scripts.recent_change_features import change_features_for_request, change_lineup_features_for_request, NEW_FEATURES


def game(day,pts=10,minutes=20,player='Player',team='AAA'):
    return dict(player_id=player,team_id=team,game_id=f'{team} @ BBB|{day}',GAMEDATE=day,
                PTS=str(pts),MIN=str(minutes),REB='4',AST='2',is_home='0',is_neutral_site='0')


class RecentChangeTests(unittest.TestCase):
    def history(self):
        return [game(f'2025-10-{d:02d}',p,m) for d,p,m in zip(range(21,26),[0,10,20,30,40],[10,20,30,40,50])]

    def test_exact_lags_mean_change_and_population_variance(self):
        out=change_features_for_request(self.history(),game('2025-10-27'))
        self.assertTrue(out['eligible'])
        self.assertEqual(out['pts_last_1'],40.)
        self.assertEqual(out['min_mean_3'],40.)
        self.assertEqual(out['min_change_last_vs_5'],20.)
        self.assertAlmostEqual(out['pts_std_5'],math.sqrt(200))
        self.assertAlmostEqual(out['min_std_5'],math.sqrt(200))

    def test_target_same_day_future_outcomes_and_order_never_change_features(self):
        h=self.history(); target=game('2025-10-27',900,1)
        expected=change_features_for_request(h,target)
        target.update(PTS='99999',MIN='99999')
        changed=change_features_for_request(list(reversed(h))+[target,game('2025-10-28',999,999)],target)
        self.assertEqual(expected,changed)

    def test_zero_minutes_excluded_but_positive_minutes_zero_points_retained(self):
        h=self.history()+[game('2025-10-26',999,0)]
        out=change_features_for_request(h,game('2025-10-27'))
        self.assertEqual(out['history_count'],5)
        self.assertEqual(out['pts_last_1'],40.)
        self.assertAlmostEqual(out['pts_std_5'],math.sqrt(200))

    def test_other_player_and_previous_season_do_not_enter_window(self):
        h=self.history()+[game('2025-10-26',999,999,player='Other'),game('2025-04-01',999,999)]
        out=change_features_for_request(h,game('2025-10-27',team='CCC'))
        self.assertEqual(out['pts_last_1'],40.)
        reset=change_features_for_request(h,game('2026-10-27'))
        self.assertFalse(reset['eligible'])
        self.assertTrue(all(reset[c] is None for c in NEW_FEATURES))

    def test_missing_recent_values_do_not_fall_back_to_older_games(self):
        h=self.history()+[game('2025-10-26')]
        h[-1]['MIN']=''
        out=change_features_for_request(h,game('2025-10-27'))
        self.assertFalse(out['eligible'])
        self.assertTrue(all(out[c] is None for c in NEW_FEATURES))

    def test_future_target_needs_no_outcome_and_constant_history_has_zero_volatility(self):
        h=[game(f'2025-11-{i:02d}',0,20) for i in range(1,6)]
        target=dict(player_id='Player',team_id='AAA',GAMEDATE='2025-11-06',is_home=0,is_neutral_site=0)
        out=change_features_for_request(h,target)
        self.assertTrue(out['eligible'])
        self.assertEqual(out['pts_std_5'],0.)
        self.assertEqual(out['min_std_5'],0.)
        self.assertEqual(out['min_change_last_vs_5'],0.)

    def test_insufficient_and_duplicate_history_are_not_silently_accepted(self):
        target=game('2025-10-27')
        out=change_features_for_request(self.history()[:4],target)
        self.assertFalse(out['eligible'])
        with self.assertRaises(ValueError):
            change_features_for_request(self.history()+[self.history()[0]],target)

    def test_full_entry_keeps_team_gate_and_does_not_claim_historical_pregame(self):
        target=game('2025-10-27')
        teams=[dict(team_id='AAA',GAMEDATE=f'2025-10-{d:02d}',game_id=f'AAA|2025-10-{d:02d}',status='completed') for d in range(21,26)]
        starters=['Player','A','B','C','D']
        good=change_lineup_features_for_request(self.history(),teams,target,starters,mode='historical_postgame')
        self.assertTrue(good['eligible'])
        self.assertFalse(good['live_prediction_ready'])
        self.assertEqual(good['pts_last_1'],40.)
        bad=change_lineup_features_for_request(self.history(),teams[:4],target,starters,mode='historical_postgame')
        self.assertFalse(bad['eligible'])
        self.assertFalse(bad['live_prediction_ready'])


if __name__=='__main__':
    unittest.main()
