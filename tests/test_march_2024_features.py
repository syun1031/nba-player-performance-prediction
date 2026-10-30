import unittest
import numpy as np
from scripts.march_2024_features import march_request_features,build_test_features
from scripts.historical_season_features import request_features as development_request
from scripts.team_lineup_features import TeamHistoryIndex


class MarchFeaturesTests(unittest.TestCase):
    def fixture(self):
        history=[dict(player_id='A',team_id='BOS',game_id=f'BOS VS. NYK|2025-02-{d}',GAMEDATE=f'2025-02-{d}',MIN=30,PTS=20,REB=5,AST=6) for d in range(22,27)]
        teams=[dict(team_id=t,opponent_id=o,game_id=str(d),GAMEDATE=f'2025-02-{d}',status='completed',points_allowed=100,combined_points=210) for d in range(22,27) for t,o in [('BOS','NYK'),('NYK','BOS')]]
        target=dict(player_id='A',team_id='BOS',nba_player_id='7',nba_game_id='g1',game_id='BOS VS. NYK|2025-03-01',GAMEDATE='2025-03-01',is_home=1,is_neutral_site=0,is_starter=1,starting_players=['A','B','C','D','E'])
        return history,teams,target,dict(season='2024-25',teams={'BOS':'East','NYK':'East'})

    def test_march_entry_preserves_development_guard(self):
        h,t,r,c=self.fixture()
        self.assertTrue(march_request_features(h,TeamHistoryIndex(t),r,{},c)['eligible'])
        with self.assertRaises(ValueError):development_request(h,TeamHistoryIndex(t),r,{},c)
        for day in ['2025-02-28','2025-04-14','2026-03-01']:
            with self.assertRaises(ValueError):march_request_features(h,TeamHistoryIndex(t),dict(r,GAMEDATE=day),{},c)

    def test_own_same_day_and_future_outcomes_do_not_change_features(self):
        h,t,r,c=self.fixture();a=march_request_features(h,TeamHistoryIndex(t),r,{},c)
        poison=[dict(h[0],GAMEDATE=d,MIN='SEALED',PTS='SEALED',REB='SEALED',AST='SEALED') for d in ['2025-03-01','2025-03-02']]
        team_poison=[dict(t[0],GAMEDATE=d,game_id=d,points_allowed='SEALED',combined_points='SEALED') for d in ['2025-03-01','2025-03-02']]
        b=march_request_features(h+poison,TeamHistoryIndex(t+team_poison),dict(r,PTS='SEALED',MIN='SEALED',target_pts='SEALED'),{},c)
        for k in ['pts_mean_5','pts_last_1','same_conf_h2h_count','same_conf_h2h_pts_mean','opp_points_allowed_mean_5']:self.assertEqual(a[k],b[k])
        self.assertTrue(np.isnan(b['prev_season_pts_mean']))

    def test_later_game_uses_completed_prior_march_but_not_zero_minutes(self):
        h,t,r,c=self.fixture()
        h+=[dict(h[0],GAMEDATE='2025-02-27',game_id='z',MIN=0,PTS=1),dict(h[0],GAMEDATE='2025-03-01',game_id=r['game_id'],MIN=30,PTS=40)]
        r=dict(r,GAMEDATE='2025-03-02',game_id='BOS VS. NYK|2025-03-02')
        a=march_request_features(h,TeamHistoryIndex(t),r,{},c)
        self.assertEqual(a['pts_mean_5'],24);self.assertEqual(a['pts_last_1'],40);self.assertEqual(a['same_conf_h2h_count'],6)


if __name__=='__main__':unittest.main()
