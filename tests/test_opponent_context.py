import unittest
from copy import deepcopy

from scripts.opponent_context import build_team_games,opponent_features,opponent_lineup_features,CONTEXT_FEATURES
from tests.test_recent_change_features import game


def logs(day='2025-10-21',a=80,b=90):
    return [dict(PLAYER=p,TEAM=t,MATCHUP=m,GAMEDATE=day,MIN=str(mi),PTS=str(s)) for p,t,m,mi,s in [
        ('A1','AAA','AAA VS. BBB',30,30),('A2','AAA','AAA VS. BBB',20,a-30),
        ('B1','BBB','BBB @ AAA',40,b),('B2','BBB','BBB @ AAA',10,0),
        ('DNP','AAA','AAA VS. BBB',0,0)]]


class OpponentContextTests(unittest.TestCase):
    def team_games(self):
        return build_team_games(sum([logs(f'2025-10-{i:02d}',80+i-21,90+i-21) for i in range(21,27)],[]))

    def target(self,day='2025-10-27'):
        return dict(player_id='Player',team_id='AAA',game_id='AAA @ BBB|'+day,GAMEDATE=day)

    def test_aggregation_pairs_both_perspectives_and_keeps_all_player_points(self):
        teams=build_team_games(logs())
        self.assertEqual(len(teams),2)
        a=next(r for r in teams if r['team_id']=='AAA')
        self.assertEqual((a['points_for'],a['points_allowed'],a['combined_points']),(80,90,170))
        self.assertEqual(a['positive_minute_players'],2)

    def test_neutral_home_direction_disagreement_does_not_create_two_games(self):
        rows=logs()
        for r in rows:
            if r['TEAM']=='BBB': r['MATCHUP']='BBB VS. AAA'
        self.assertEqual(len(build_team_games(rows)),2)

    def test_missing_other_side_duplicate_or_bad_outcome_is_rejected(self):
        for bad in [logs()[:2],logs()+logs()[:1],logs()+[dict(logs()[0],PLAYER='Another',PTS='nan')],
                    logs()+[dict(logs()[-1],PLAYER='Bad DNP',PTS='1')]]:
            with self.assertRaises(ValueError):
                build_team_games(bad)

    def test_future_raw_results_are_skipped_before_reading_outcomes(self):
        class Hidden(dict):
            def __getitem__(self,k):
                if k in ['MIN','PTS']: raise AssertionError('Consumed test outcomes accessed')
                return super().__getitem__(k)
        self.assertEqual(build_team_games(logs()+[Hidden(dict(logs()[0],GAMEDATE='2026-03-01'))]),build_team_games(logs()))

    def test_exact_recent_five_opponent_points_allowed_and_combined(self):
        out=opponent_features(self.team_games(),self.target())
        self.assertTrue(out['opponent_eligible'])
        self.assertEqual(out['opponent_id'],'BBB')
        self.assertEqual(out['opp_points_allowed_mean_5'],83.)
        self.assertEqual(out['opp_combined_points_mean_5'],176.)
        self.assertEqual(len(out['opponent_history_game_ids_5']),5)

    def test_same_day_future_and_current_outcomes_never_affect_features(self):
        teams=self.team_games(); expected=opponent_features(teams,self.target())
        future=build_team_games(logs('2025-10-27',900,999)+logs('2025-10-28',500,600))
        self.assertEqual(expected,opponent_features(list(reversed(teams))+future,dict(self.target(),PTS=999,MIN=999)))

    def test_opponent_five_game_gate_and_season_reset(self):
        teams=self.team_games()
        for target in [self.target('2025-10-25'),self.target('2026-10-27')]:
            out=opponent_features(teams,target)
            self.assertFalse(out['opponent_eligible'])
            self.assertTrue(all(out[c] is None for c in CONTEXT_FEATURES))

    def test_duplicate_history_and_missing_recent_values_never_fall_back(self):
        teams=self.team_games()
        with self.assertRaises(ValueError):
            opponent_features(teams+teams[:1],self.target())
        bad=deepcopy(teams)
        next(r for r in bad if r['team_id']=='BBB' and r['GAMEDATE']=='2025-10-26')['points_allowed']=None
        with self.assertRaises(ValueError):
            opponent_features(bad,self.target())

    def test_source_roster_mismatch_is_rejected(self):
        fake={('2025-10-21',t):dict(players=['Wrong'],pair=frozenset(['AAA','BBB'])) for t in ['AAA','BBB']}
        with self.assertRaises(ValueError):
            build_team_games(logs(),fake)

    def test_two_alias_spellings_of_same_player_cannot_double_count_points(self):
        rows=logs()
        rows[0]['PLAYER']='Jonas Valančiūnas'
        rows[1]['PLAYER']='Jonas Valanciunas'
        with self.assertRaises(ValueError):
            build_team_games(rows)

    def test_full_entry_retains_personal_team_and_lineup_gates(self):
        history=[game(f'2025-10-{d:02d}') for d in range(21,26)]
        target=game('2025-10-27'); starters=['Player','A','B','C','D']
        out=opponent_lineup_features(history,self.team_games(),target,starters,mode='historical_postgame')
        self.assertTrue(out['eligible']); self.assertFalse(out['live_prediction_ready'])
        out=opponent_lineup_features(history[:4],self.team_games(),target,starters,mode='historical_postgame')
        self.assertFalse(out['eligible'])
        out=opponent_lineup_features(history,self.team_games(),target,starters[:4],mode='historical_postgame')
        self.assertFalse(out['eligible'])


if __name__=='__main__':
    unittest.main()
