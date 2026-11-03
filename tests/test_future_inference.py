import copy
from datetime import timedelta
import unittest
from pathlib import Path
from tests.test_future_source import fixture,NOW
from nba_dashboard.future_source import assess_context
from nba_dashboard.future_inference import build_features,route_prediction,prediction_csv_row
from nba_dashboard.future_data import history_issues,load_future,model_minutes
from nba_dashboard.future_inference import load_models,predict_features


def history_fixture():
    rows=[];teams=[]
    for i in range(6):
        day=f'2026-10-{25+i:02}'
        for code,opp in [('WAS','ATL'),('ATL','WAS')]:
            gid=f'{code} VS. {opp}|{day}'
            teams.append(dict(team_id=code,opponent_id=opp,GAMEDATE=day,game_id=gid,
                points_for=100,points_allowed=110,combined_points=210,status='completed'))
        rows.append(dict(nba_player_id='100',player_id='WAS 0',PLAYER='WAS 0',TEAM='WAS',team_id='WAS',
            GAMEDATE=day,game_id=f'WAS VS. ATL|{day}',MATCHUP='WAS VS. ATL',MIN=30,PTS=20,REB=5,AST=4))
    return dict(history=rows,raw_history=copy.deepcopy(rows),team_games=teams,profiles={},
        conferences=dict(season='2026-27',teams={'WAS':'East','ATL':'East'}),
        manifest=dict(through='2026-11-03',collected_at_utc='2026-11-04T23:00:00Z',version='fixture'),pending=[])


class FutureInferenceTests(unittest.TestCase):
    def test_full_25_features_no_prior_and_bench(self):
        c=assess_context(*fixture(),NOW);b=history_fixture();r=build_features(b,c,'100',NOW)
        self.assertTrue(r['eligible'],r['exclusion_reason'])
        self.assertEqual(r['pts_mean_5'],20)
        self.assertEqual(r['same_conf_h2h_count'],6)
        self.assertEqual(r['has_prev_season'],0)
        self.assertEqual(r['is_starter'],1)
        # Confirmed active bench member has no personal history, not an inferred DNP.
        r=build_features(b,c,'105',NOW)
        self.assertEqual(r['is_starter'],0)
        self.assertFalse(r['eligible'])

    def test_player_team_opponent_five_gate(self):
        for kind in ['personal','team','opponent']:
            b=history_fixture();c=assess_context(*fixture(),NOW)
            if kind=='personal':b['history']=b['history'][:4]
            else:
                code='WAS' if kind=='team' else 'ATL'
                b['team_games']=[r for i,r in enumerate(b['team_games']) if r['team_id']!=code or i<8]
            with self.subTest(kind=kind):self.assertFalse(build_features(b,c,'100',NOW)['eligible'])

    def test_same_day_future_and_other_season_do_not_change_features(self):
        b=history_fixture();c=assess_context(*fixture(),NOW)
        first=build_features(b,c,'100',NOW)
        for day in ['2026-11-04','2026-11-05','2025-10-25']:
            b['history'].append(dict(b['history'][0],GAMEDATE=day,game_id='WAS VS. ATL|'+day,PTS=9999,MIN=9999))
        second=build_features(b,c,'100',NOW)
        for key in ['pts_mean_5','min_mean_5','same_conf_h2h_count','same_conf_h2h_pts_mean']:
            self.assertEqual(first[key],second[key])

    def test_missing_history_coverage_and_pending_prior_game_block(self):
        b=history_fixture();self.assertEqual(history_issues(b,'2026-11-04',NOW),[])
        b['manifest']['through']='2026-11-02'
        self.assertIn('history_not_updated',history_issues(b,'2026-11-04',NOW))
        b=history_fixture();b['pending']=[dict(GAMEDATE='2026-10-27')]
        self.assertIn('unresolved_prior_games',history_issues(b,'2026-11-04',NOW))

    def test_inactive_and_expired_confirmation_cannot_predict(self):
        c=assess_context(*fixture(),NOW);b=history_fixture()
        for pid,now in [('106',NOW),('100',NOW+timedelta(minutes=20))]:
            self.assertFalse(build_features(b,c,pid,now)['eligible'])

    def test_unknown_official_id_cannot_borrow_same_name_history(self):
        s,box,lineup=fixture()
        lineup['data']['games'][0]['homeTeam']['players'][0]['personId']=999
        c=assess_context(s,box,lineup,NOW)
        f=build_features(history_fixture(),c,'999',NOW)
        self.assertFalse(f['eligible'])
        self.assertEqual(f['history_count'],0)

    def test_route_inclusive_25_and_no_target_labels(self):
        self.assertEqual(route_prediction(24.99,22,28)['model_variant'],'combined_prior25_w2')
        self.assertEqual(route_prediction(25,22,28)['prediction'],28)
        self.assertEqual(route_prediction(25,22,28)['model_variant'],'residual25_w2')
        with self.assertRaises(ValueError):route_prediction(float('nan'),22,28)

    def test_minutes_match_integer_training_and_preserve_source(self):
        r=dict(minutes_source='11:30',MIN=11.5)
        self.assertEqual(model_minutes(r),12)
        self.assertEqual(r['MIN'],11.5)
        self.assertEqual(model_minutes(dict(minutes_source='00:29',MIN=29/60)),0)

    def test_csv_keeps_source_times_model_hashes_and_history_counts(self):
        _,box,lineup=fixture()
        r=dict(prediction=22,features={'pts_mean_5':20},history_counts={'history_count':6},
            models={'combined_prior25_w2':{'sha256':'b'*64,'trained_through':'2026-01-31'}},
            sources=[box,lineup],schedule_source={'url':'https://www.nba.com/games?date=2026-11-04','observed_at_utc':NOW.isoformat(),'sha256':'c'*64})
        flat=prediction_csv_row(r)
        self.assertEqual(flat['box_observed_at_utc'],NOW.isoformat())
        self.assertEqual(flat['lineup_source_url'],lineup['url'])
        self.assertEqual(flat['model_combined_prior25_w2_sha256'],'b'*64)
        self.assertEqual(flat['history_count'],6)
        self.assertEqual(flat['schedule_sha256'],'c'*64)

    def test_frozen_models_reproduce_saved_march_predictions(self):
        import csv,math
        root=Path(__file__).resolve().parents[1]
        bundle=load_future(root);models=load_models(root,bundle)
        with (root/'outputs/season_test_2025_26_prior_models/features.csv').open(encoding='utf-8-sig') as f:features=list(csv.DictReader(f))[:4]
        with (root/'outputs/dashboard_comparison/2025_26_test.csv').open(encoding='utf-8-sig') as f:saved={(r['player_id'],r['game_id']):r for r in csv.DictReader(f)}
        for r in features:
            r['pts_mean_5']=float(r['pts_mean_5']);p=predict_features(r,models);old=saved[r['player_id'],r['game_id']]
            self.assertAlmostEqual(p['direct_prediction'],float(old['pred_combined_prior25_w2']),places=10)
            self.assertAlmostEqual(p['offset_prediction'],float(old['pred_residual25_w2']),places=10)


if __name__=='__main__':unittest.main()
