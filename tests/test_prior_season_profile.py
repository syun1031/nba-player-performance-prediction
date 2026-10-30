import copy
import unittest
import numpy as np
from scripts.prior_season_profile import build_profiles,augment,profile_matrix,PROFILE_FEATURES


def game(name='A',day='2024-11-01',team='AAA',minutes=10,points=5):
    return dict(PLAYER=name,TEAM=team,MATCHUP=team+' VS. BBB',GAMEDATE=day,MIN=minutes,PTS=points)


class PriorSeasonProfileTests(unittest.TestCase):
    def test_transfer_aggregates_games_and_rate_is_ratio_of_totals(self):
        p=build_profiles([game(minutes=10,points=10),game(day='2025-01-01',team='CCC',minutes=30,points=0)])['A']
        self.assertEqual([p[k] for k in PROFILE_FEATURES],[5,20,.25,2,1])

    def test_zero_minutes_excluded_zero_points_kept(self):
        p=build_profiles([game(minutes=0,points=0),game(day='2024-12-01',minutes=12,points=0)])
        self.assertEqual(p['A']['prev_season_games'],1)
        self.assertEqual(p['A']['prev_season_pts_mean'],0)
        self.assertEqual(build_profiles([game(minutes=0)]),{})

    def test_no_prior_keeps_player_and_uses_nan_not_zero(self):
        rows=[dict(player_id='New',GAMEDATE='2025-11-01',target_pts=40)]
        result=augment(rows,{})
        self.assertEqual(len(result),1)
        self.assertTrue(all(np.isnan(result[0][k]) for k in PROFILE_FEATURES[:3]))
        self.assertEqual([result[0][k] for k in PROFILE_FEATURES[3:]],[0,0])
        self.assertNotIn('has_prev_season',rows[0])

    def test_exact_name_only_no_guessing(self):
        profiles=build_profiles([game(name='José')])
        r=augment([dict(player_id='Jose',GAMEDATE='2025-11-01')],profiles)[0]
        self.assertEqual(r['has_prev_season'],0)

    def test_reject_wrong_season_duplicate_nonfinite_and_negative(self):
        for rows in [[game(day='2025-10-01')],[game(day='2024-01-01')],[game(),game()],
                     [game(minutes=-1)],[game(points=-1)],[game(points=float('nan'))],[game(minutes=float('inf'))]]:
            with self.subTest(rows=rows),self.assertRaises(ValueError): build_profiles(rows)

    def test_augment_dates_and_existing_columns_protected(self):
        for row in [dict(player_id='A',GAMEDATE='2025-04-12'),dict(player_id='A',GAMEDATE='2026-03-01'),
                    dict(player_id='A',GAMEDATE='2025-11-01',has_prev_season=1)]:
            with self.assertRaises(ValueError): augment([row],{})

    def test_feature_whitelist_does_not_read_labels_and_missing_is_preserved(self):
        rows=augment([dict(player_id='A',GAMEDATE='2025-11-01',is_home=1)],{})
        saved=copy.deepcopy(rows)
        a=profile_matrix(rows,['is_home']+PROFILE_FEATURES)
        b=profile_matrix([dict(r,target_pts=99999,PTS=99999,MIN=99999) for r in rows],['is_home']+PROFILE_FEATURES)
        np.testing.assert_equal(a,b)
        self.assertEqual(rows[0]['is_home'],saved[0]['is_home'])
        self.assertEqual(a.shape,(1,6))
        with self.assertRaises(ValueError): profile_matrix(rows,['target_pts'])


if __name__=='__main__': unittest.main()
