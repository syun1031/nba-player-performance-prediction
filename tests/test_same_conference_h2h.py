import unittest
import math
from copy import deepcopy

from scripts.same_conference_h2h import h2h_features,h2h_lineup_features,h2h_matrix,H2H_FEATURES,CONFERENCES


def history_row(day,points=10,minutes=20,team='LAL',opponent='DEN',player='P'):
    return dict(player_id=player,team_id=team,game_id=f'{team} @ {opponent}|{day}',GAMEDATE=day,
                PTS=str(points),MIN=str(minutes),REB='4',AST='2')


class SameConferenceH2HTests(unittest.TestCase):
    def target(self,team='LAL',opponent='DEN',day='2025-10-27'):
        return dict(player_id='P',team_id=team,game_id=f'{team} @ {opponent}|{day}',GAMEDATE=day,
                    is_home=0,is_neutral_site=0)

    def history(self):
        return [history_row('2025-10-21',10),history_row('2025-10-22',0),
                history_row('2025-10-23',30,opponent='SAC'),history_row('2025-10-24',0,0),
                history_row('2025-10-25',20),history_row('2025-10-26',40,opponent='MIN')]

    def test_conference_snapshot_has_thirty_teams_with_fifteen_each(self):
        self.assertEqual(len(CONFERENCES['teams']),30)
        self.assertEqual(list(CONFERENCES['teams'].values()).count('East'),15)
        self.assertEqual(list(CONFERENCES['teams'].values()).count('West'),15)

    def test_last_and_mean_use_same_opponent_and_keep_zero_point_appearances(self):
        out=h2h_features(self.history(),self.target())
        self.assertEqual(out['same_conference'],1)
        self.assertEqual(out['same_conf_h2h_count'],3)
        self.assertEqual(out['same_conf_h2h_pts_last_1'],20)
        self.assertEqual(out['same_conf_h2h_pts_mean'],10)
        self.assertEqual(out['h2h_status'],'available')
        self.assertEqual(out['h2h_last_history_date'],'2025-10-25')

    def test_target_same_day_future_and_other_player_outcomes_are_ignored(self):
        target=self.target(); old=h2h_features(self.history(),target)
        more=[history_row('2025-10-27',999),history_row('2025-10-28',999),history_row('2025-10-26',999,player='Other')]
        self.assertEqual(old,h2h_features(list(reversed(self.history()))+more,dict(target,PTS=999,MIN=999)))

    def test_no_prior_meeting_and_cross_conference_are_missing_not_zero_points(self):
        first=h2h_features([],self.target())
        cross=h2h_features([history_row('2025-10-21',80,team='LAL',opponent='BOS')],self.target(opponent='BOS'))
        for row in [first,cross]:
            self.assertEqual(row['same_conf_h2h_count'],0)
            self.assertIsNone(row['same_conf_h2h_pts_last_1']); self.assertIsNone(row['same_conf_h2h_pts_mean'])
        self.assertEqual(first['same_conference'],1); self.assertEqual(cross['same_conference'],0)
        self.assertEqual(cross['h2h_history_game_ids'],[])

    def test_transfer_preserves_personal_opponent_history_and_prior_season_resets(self):
        out=h2h_features([history_row('2025-10-21',30,team='BOS'),history_row('2025-04-01',99)],self.target())
        self.assertEqual(out['same_conf_h2h_count'],1)
        self.assertEqual(out['same_conf_h2h_pts_mean'],30)

    def test_unknown_teams_unverified_season_and_target_date_mismatch_are_rejected(self):
        for target in [self.target(team='XXX'),self.target(day='2026-10-27'),dict(self.target(),GAMEDATE='2025-10-26')]:
            with self.assertRaises(ValueError): h2h_features([],target)

    def test_missing_values_and_duplicate_meetings_do_not_fall_back(self):
        for history in [self.history()+[self.history()[0]],
                        self.history()+[dict(history_row('2025-10-26'),PTS='')],
                        self.history()+[dict(history_row('2025-10-26'),MIN='nan')]]:
            with self.assertRaises(ValueError): h2h_features(history,self.target())

    def test_actual_zero_point_meeting_and_more_than_four_meetings_are_valid(self):
        history=[history_row(f'2025-10-{d:02d}',0) for d in range(21,27)]
        out=h2h_features(history,self.target())
        self.assertEqual(out['same_conf_h2h_count'],6)
        self.assertEqual(out['same_conf_h2h_pts_mean'],0)
        self.assertEqual(out['same_conf_h2h_pts_last_1'],0)

    def test_matrix_allows_only_intentional_missing_score_columns(self):
        first=h2h_features([],self.target())
        x=h2h_matrix([dict(first,pts_mean_5=10)],['pts_mean_5']+H2H_FEATURES)
        self.assertTrue(math.isnan(x[0,-1])); self.assertTrue(math.isnan(x[0,-2]))
        self.assertEqual(x[0,2],0)
        for row in [dict(first,pts_mean_5=float('nan')),dict(first,pts_mean_5=10,same_conf_h2h_count=1),
                    dict(first,pts_mean_5=10,same_conf_h2h_pts_mean=0)]:
            with self.assertRaises(ValueError): h2h_matrix([row],['pts_mean_5']+H2H_FEATURES)
        with self.assertRaises(ValueError): h2h_matrix([dict(first,target_pts=10)],['target_pts']+H2H_FEATURES)

    def test_full_entry_retains_team_player_lineup_and_opponent_gates(self):
        teams=[dict(team_id=t,GAMEDATE=f'2025-10-{d:02d}',game_id=f'{t}|{d}',status='completed',
                    points_allowed=100,combined_points=210) for t in ['LAL','DEN'] for d in range(21,27)]
        starters=['P','A','B','C','D']
        good=h2h_lineup_features(self.history(),teams,self.target(),starters,mode='historical_postgame')
        self.assertTrue(good['eligible']); self.assertFalse(good['live_prediction_ready'])
        bad=h2h_lineup_features(self.history()[:3],teams,self.target(),starters,mode='historical_postgame')
        self.assertFalse(bad['eligible'])
        bad=h2h_lineup_features(self.history(),teams,self.target(),starters[:4],mode='historical_postgame')
        self.assertFalse(bad['eligible'])
