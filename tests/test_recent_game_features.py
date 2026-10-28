import math
import unittest
from copy import deepcopy
from scripts.recent_game_features import features_for_request, build_feature_rows, split_for_date
from scripts.build_features_baseline import regression_metrics


def game(day, pts=10, minutes=20, team='AAA', player='Player'):
    return dict(player_id=player, team_id=team, game_id=f'{team} @ BBB|{day}',
                GAMEDATE=day, MIN=str(minutes), PTS=str(pts), REB='4', AST='2',
                is_home='0', is_neutral_site='0', roster_match_status='matched')


class FeatureTests(unittest.TestCase):
    def history(self):
        return [game(f'2025-10-{d:02d}', p) for d,p in zip(range(21,26),[10,20,0,30,40])]

    def test_actual_and_future_results_not_used(self):
        history=self.history()
        target=game('2025-10-27',99)
        a=features_for_request(history+[target,game('2025-10-28',100)],target)
        target['PTS']='9999'
        b=features_for_request(list(reversed(history))+[target,game('2025-10-28',0)],target)
        self.assertEqual(a,b)
        self.assertEqual(a['pts_mean_5'],20)
        self.assertEqual(a['pts_mean_10'],20)
        self.assertEqual(a['rest_days'],1)

    def test_zero_minutes_ignored_and_zero_points_kept(self):
        history=self.history()+[game('2025-10-26',100,minutes=0)]
        f=features_for_request(history,game('2025-10-27'))
        self.assertEqual(f['history_count'],5)
        self.assertEqual(f['pts_mean_5'],20)
        self.assertEqual(f['last_history_date'],'2025-10-25')

    def test_transfer_preserves_history(self):
        f=features_for_request(self.history(),game('2025-10-27',team='CCC'))
        self.assertTrue(f['eligible'])
        self.assertEqual(f['history_count'],5)

    def test_season_resets_and_other_player_excluded(self):
        history=self.history()+[game('2025-04-01',90),game('2025-10-26',80,player='Other')]
        f=features_for_request(history,game('2025-10-27'))
        self.assertEqual(f['history_count'],5)
        self.assertEqual(features_for_request(self.history(),game('2026-10-27'))['exclusion_reason'],'insufficient_history')

    def test_insufficient_history_and_partial_ten_window(self):
        self.assertFalse(features_for_request(self.history()[:4],game('2025-10-27'))['eligible'])
        f=features_for_request(self.history()+[game('2025-10-26',60)],game('2025-10-27'))
        self.assertEqual(f['n_10'],6)
        self.assertAlmostEqual(f['pts_mean_10'],160/6)

    def test_missing_recent_stat_not_silently_skipped(self):
        history=self.history()+[game('2025-10-26')]
        history[-1]['PTS']=''
        f=features_for_request(history,game('2025-10-27'))
        self.assertFalse(f['eligible'])
        self.assertIn('invalid_history_pts',f['exclusion_reason'])

    def test_same_day_history_excluded_and_duplicate_rejected(self):
        h=self.history()+[game('2025-10-27',500)]
        f=features_for_request(h,game('2025-10-27'))
        self.assertEqual(f['history_count'],5)
        with self.assertRaises(ValueError):
            build_feature_rows(self.history()+[self.history()[0]])

    def test_batch_order_invariant_and_zero_row_not_sample(self):
        records=self.history()+[game('2025-10-26',100,minutes=0),game('2025-10-27')]
        a=build_feature_rows(records)
        b=build_feature_rows(list(reversed(records)))
        self.assertEqual(a,b)
        self.assertEqual(len(a),6)
        self.assertEqual(a[-1]['pts_mean_5'],20)

    def test_month_split_boundaries(self):
        for d,s in [('2026-01-31','train'),('2026-02-01','validation'),
                    ('2026-02-28','validation'),('2026-03-01','test'),('2026-04-12','test')]:
            self.assertEqual(split_for_date(d),s)
        with self.assertRaises(ValueError):
            split_for_date('2026-04-13')

    def test_noncanonical_dates_must_not_be_lexically_misordered(self):
        with self.assertRaises(ValueError):
            split_for_date('20260131')

    def test_window_moves_forward_and_future_request_has_no_stats(self):
        history=[game(f'2025-11-{i:02d}',i) for i in range(1,11)]
        target=dict(player_id='Player',GAMEDATE='2025-11-11',is_home=0,is_neutral_site=0)
        f=features_for_request(history,target)
        self.assertEqual(f['pts_mean_5'],8)
        self.assertEqual(f['pts_mean_10'],5.5)
        self.assertTrue(f['eligible'])

    def test_metrics_hand_calculation_and_empty(self):
        m=regression_metrics([0,10],[2,6])
        self.assertEqual(m['mae'],3)
        self.assertAlmostEqual(m['rmse'],math.sqrt(10))
        self.assertIsNone(regression_metrics([],[])['mae'])


if __name__=='__main__':
    unittest.main()
