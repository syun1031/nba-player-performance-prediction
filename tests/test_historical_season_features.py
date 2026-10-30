import unittest
import numpy as np
from scripts.historical_season_features import previous_profiles,request_features,season_folds,development_rows
from scripts.team_lineup_features import TeamHistoryIndex


class HistoricalSeasonFeaturesTests(unittest.TestCase):
    def test_profile_uses_stable_id_and_ratio_totals(self):
        rows=[dict(nba_player_id='7',GAMEDATE='2023-11-01',MIN=10,PTS=10),dict(nba_player_id='7',GAMEDATE='2024-01-01',MIN=30,PTS=0)]
        p=previous_profiles(rows)['7'];self.assertEqual(p['prev_season_pts_mean'],5);self.assertEqual(p['prev_season_pts_per_min'],.25)
        with self.assertRaises(ValueError):previous_profiles([dict(rows[0],GAMEDATE='2024-11-01')])

    def test_request_ignores_target_and_future_outcomes_and_keeps_missing(self):
        history=[dict(player_id='Player',team_id='BOS',game_id=f'BOS VS. NYK|2024-10-{d}',GAMEDATE=f'2024-10-{d}',MIN=30,PTS=20,REB=5,AST=6) for d in range(22,27)]
        teams=[dict(team_id=t,opponent_id=o,game_id=str(d),GAMEDATE=f'2024-10-{d}',status='completed',points_allowed=100,combined_points=210) for d in range(22,27) for t,o in [('BOS','NYK'),('NYK','BOS')]]
        target=dict(player_id='Player',team_id='BOS',nba_player_id='7',game_id='BOS VS. NYK|2024-10-27',GAMEDATE='2024-10-27',is_home=1,is_neutral_site=0,starting_players=['Player','B','C','D','E'])
        conf=dict(season='2024-25',teams={'BOS':'East','NYK':'East'})
        a=request_features(history,TeamHistoryIndex(teams),target,{},conf)
        b=request_features(history+[dict(history[0],GAMEDATE='2024-11-01',PTS='SEALED')],TeamHistoryIndex(teams),dict(target,PTS='SEALED',MIN='SEALED',target_pts='SEALED'),{},conf)
        self.assertTrue(a['eligible']);self.assertEqual(a['pts_mean_5'],20);self.assertEqual(a['same_conf_h2h_count'],5)
        self.assertTrue(np.isnan(a['prev_season_pts_mean']));self.assertEqual(a['has_prev_season'],0)
        for key in ['pts_mean_5','same_conf_h2h_count','opp_points_allowed_mean_5','is_starter']:self.assertEqual(a[key],b[key])
        with self.assertRaises(ValueError):request_features(history,TeamHistoryIndex(teams),target,{},dict(conf,season='2025-26'))

    def test_development_filter_precedes_reading_test_labels(self):
        rows=[dict(GAMEDATE='2025-03-01',MIN='SEALED',PTS='SEALED'),dict(GAMEDATE='2024-10-22',MIN=10,PTS=0)]
        self.assertEqual(len(development_rows(rows)),1)

    def test_fold_dates_shift_one_year_and_do_not_include_february(self):
        rows=[dict(GAMEDATE=d,player_id='A',game_id=d) for d in ['2024-11-30','2024-12-01','2024-12-31','2025-01-01','2025-01-15','2025-01-16','2025-01-31']]
        folds=season_folds(rows);self.assertEqual([len(va) for _,_,va in folds],[2,2,2])
        with self.assertRaises(ValueError):season_folds(rows+[dict(GAMEDATE='2025-02-01',player_id='A',game_id='x')])

    def test_complete_pipeline_excludes_zero_minute_points_and_sealed_march(self):
        from scripts.experiment_2024_season import prepare_features
        records=[];teams=[]
        for d in range(22,28):
            day=f'2024-10-{d}'
            for t,o in [('BOS','NYK'),('NYK','BOS')]:
                records.append(dict(player_id=t+' P',team_id=t,nba_player_id=t,nba_game_id=str(d),game_id=t+' VS. '+o+'|'+day,
                    GAMEDATE=day,MIN=30,PTS=20,REB=5,AST=6,is_home=1,is_neutral_site=0,is_starter=1,starting_players=[t+' P','B','C','D','E']))
                teams.append(dict(team_id=t,opponent_id=o,game_id=str(d),GAMEDATE=day,status='completed',points_allowed=101,combined_points=210))
        records.append(dict(records[0],GAMEDATE='2024-10-28',game_id='BOS VS. NYK|2024-10-28',MIN=0,PTS=1))
        records.append(dict(GAMEDATE='2025-03-01',MIN='SEALED',PTS='SEALED'))
        teams.append(dict(GAMEDATE='2025-03-01',points_allowed='SEALED',combined_points='SEALED'))
        features,audit=prepare_features(records,teams,{},dict(season='2024-25',teams={'BOS':'East','NYK':'East'}))
        self.assertEqual(len(features),2);self.assertEqual(len(audit),13)
        self.assertEqual(audit[-1]['reason'],'zero_minutes')
        self.assertTrue(all(r['history_count']==5 and r['target_pts']==20 and r['opp_points_allowed_mean_5']==101 for r in features))


if __name__=='__main__':unittest.main()
