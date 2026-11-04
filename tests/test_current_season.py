import copy
import unittest
from tests.test_future_inference import history_fixture
from nba_dashboard.current_season import historical_features,backtest_rows
from tests.test_future_source import fixture,NOW
from nba_dashboard.future_source import assess_context
from nba_dashboard.future_inference import build_features
from scripts.offset_prior_model import COLUMNS


def target(bundle):
    return dict(bundle['history'][-1],GAMEDATE='2026-11-04',game_id='WAS VS. ATL|2026-11-04',
                nba_game_id='0022600173',nba_player_id='100',player_id='WAS 0',TEAM='WAS',team_id='WAS',
                MATCHUP='WAS VS. ATL',is_starter='1',is_home='1',is_neutral_site='0',PTS=99)


class CurrentSeasonTests(unittest.TestCase):
    def test_five_game_gate_and_target_outcome_isolation(self):
        b=history_fixture();t=target(b)
        f=historical_features(b,t)
        self.assertTrue(f['eligible']);self.assertEqual(f['history_count'],6)
        self.assertEqual(f['pts_mean_5'],20)
        self.assertEqual(f['lineup_source_kind'],'historical_postgame')
        self.assertFalse(f['pregame_verified'])
        t.update(PTS=500,MIN=1,REB=100,AST=100)
        self.assertEqual(f,historical_features(b,t))
        b['history']=b['history'][:4]
        self.assertIn('insufficient_history',historical_features(b,t)['exclusion_reason'])

    def test_ids_and_future_rows_cannot_supply_history(self):
        b=history_fixture();t=target(b);f=historical_features(b,t)
        later=dict(b['history'][0],GAMEDATE=t['GAMEDATE'],PTS=999,game_id=t['game_id'])
        b['history'].append(later)
        self.assertEqual(f,historical_features(b,t))
        t['nba_player_id']='999'
        self.assertIn('insufficient_history',historical_features(b,t)['exclusion_reason'])

    def test_pending_prior_game_blocks_but_later_does_not(self):
        b=history_fixture();t=target(b)
        b['pending']=[{'GAMEDATE':'2026-11-05'}]
        self.assertTrue(historical_features(b,t)['eligible'])
        b['pending']=[{'GAMEDATE':'2026-11-03'}]
        self.assertIn('unresolved_prior_games',historical_features(b,t)['exclusion_reason'])

    def test_empty_backtest_is_valid(self):
        b=history_fixture();b['raw_history']=[]
        predictions,excluded=backtest_rows(b,{})
        self.assertEqual((predictions,excluded),([],[]))

    def test_matches_live_features_without_bypassing_live_gate(self):
        b=history_fixture();t=target(b)
        live=build_features(b,assess_context(*fixture(),NOW),'100',NOW)
        historical=historical_features(b,t)
        for key in COLUMNS:
            with self.subTest(feature=key):self.assertEqual(historical[key],live[key])

    def test_team_and_opponent_five_game_boundary(self):
        for code,reason in [('WAS','insufficient_team_history'),('ATL','insufficient_opponent_history')]:
            b=history_fixture();t=target(b)
            b['team_games']=[r for r in b['team_games'] if r['team_id']!=code or r['GAMEDATE']<'2026-10-29']
            self.assertIn(reason,historical_features(b,t)['exclusion_reason'])

    def test_rounded_zero_target_excluded_before_inference(self):
        b=history_fixture();t=target(b);t.update(MIN=29/60,minutes_source='00:29')
        b['raw_history']=[t]
        predictions,excluded=backtest_rows(b,{})
        self.assertEqual(predictions,[])
        self.assertEqual(excluded[0]['reason'],'rounded_zero_minutes')
