import unittest
import scripts.team_lineup_features as module
from scripts.team_lineup_features import lineup_for_request


def games(n=5,team='AAA'):
    return [dict(team_id=team,GAMEDATE=f'2025-11-{i:02d}',game_id=f'{team}|{i}',status='completed')
            for i in range(1,n+1)]


def target(player='A',team='AAA'):
    return dict(player_id=player,team_id=team,GAMEDATE='2025-11-07',game_id='AAA @ BBB|2025-11-07')


class TeamLineupTests(unittest.TestCase):
    def historical(self,history=None,t=None,players=None):
        return lineup_for_request(games() if history is None else history,
                                  target() if t is None else t,
                                  ['A','B','C','D','E'] if players is None else players,
                                  mode='historical_postgame')

    def test_four_games_not_enough_five_is_enough(self):
        a=self.historical(games(4)); b=self.historical(games(5))
        self.assertFalse(a['lineup_eligible'])
        self.assertIn('insufficient_team_history',a['lineup_exclusion_reason'])
        self.assertTrue(b['lineup_eligible'])
        self.assertEqual(b['team_history_count'],5)

    def test_target_and_future_days_and_unfinished_not_counted(self):
        h=games()+[dict(team_id='AAA',GAMEDATE=d,game_id=d,status=s)
                   for d,s in [('2025-11-07','completed'),('2025-11-08','completed'),('2025-11-06','scheduled')]]
        self.assertEqual(self.historical(h)['team_history_count'],5)

    def test_season_reset_and_current_team_selected(self):
        h=games(team='OLD')+games(4)+[dict(team_id='AAA',GAMEDATE='2025-04-01',game_id='old',status='completed')]
        f=self.historical(h)
        self.assertEqual(f['team_history_count'],4)
        self.assertFalse(f['lineup_eligible'])

    def test_starter_has_four_teammates_and_bench_has_five(self):
        a=self.historical(); b=self.historical(t=target('Reserve'))
        self.assertEqual(a['is_starter'],1)
        self.assertEqual(a['starting_teammates'],['B','C','D','E'])
        self.assertEqual(b['is_starter'],0)
        self.assertEqual(len(b['starting_teammates']),5)
        self.assertTrue(b['lineup_eligible'])

    def test_missing_and_duplicate_lineup_not_guessed(self):
        for players in [[],['A','B','C','D'],['A','A','C','D','E']]:
            f=self.historical(players=players)
            self.assertFalse(f['lineup_eligible'])
            self.assertIsNone(f['is_starter'])

    def test_duplicate_team_date_rejected(self):
        with self.assertRaises(ValueError):
            self.historical(games()+[games()[0]])

    def test_historical_not_live_ready_and_order_invariant(self):
        a=self.historical(); b=self.historical(list(reversed(games())),players=['E','D','C','B','A'])
        self.assertEqual(a,b)
        self.assertFalse(a['pregame_verified'])
        self.assertFalse(a['live_prediction_ready'])

    def test_future_requires_time_and_confirmation(self):
        f=lineup_for_request(games(),target(),['A','B','C','D','E'])
        self.assertFalse(f['lineup_eligible'])
        self.assertIn('missing_pregame_evidence',f['lineup_exclusion_reason'])

    def test_confirmed_future_lineup_before_start_is_ready(self):
        f=lineup_for_request(games(),target(),['A','B','C','D','E'],
            confirmed=True,observed_at='2025-11-07T11:30:00+00:00',
            prediction_at='2025-11-07T11:35:00+00:00',scheduled_start_at='2025-11-07T12:00:00+00:00')
        self.assertTrue(f['live_prediction_ready'])
        self.assertTrue(f['pregame_verified'])

    def test_future_time_reversal_or_started_game_not_allowed(self):
        for observed,prediction,start in [
            ('2025-11-07T11:40:00+00:00','2025-11-07T11:35:00+00:00','2025-11-07T12:00:00+00:00'),
            ('2025-11-07T11:30:00+00:00','2025-11-07T12:00:00+00:00','2025-11-07T12:00:00+00:00')]:
            f=lineup_for_request(games(),target(),['A','B','C','D','E'],confirmed=True,
                observed_at=observed,prediction_at=prediction,scheduled_start_at=start)
            self.assertFalse(f['lineup_eligible'])

    def test_combined_future_request_requires_personal_history(self):
        person=[dict(player_id='A',GAMEDATE=f'2025-11-{i:02d}',game_id=str(i),
                     MIN='20',PTS='10',REB='4',AST='2') for i in range(1,5)]
        t=dict(target(),is_home=0,is_neutral_site=0)
        f=module.features_for_lineup_request(person,games(),t,['A','B','C','D','E'],
            confirmed=True,observed_at='2025-11-07T11:30:00+00:00',
            prediction_at='2025-11-07T11:35:00+00:00',scheduled_start_at='2025-11-07T12:00:00+00:00')
        self.assertFalse(f['live_prediction_ready'])
        self.assertFalse(f['eligible'])
        self.assertIn('insufficient_history',f['exclusion_reason'])

    def test_combined_future_request_with_both_histories_is_ready(self):
        person=[dict(player_id='A',GAMEDATE=f'2025-11-{i:02d}',game_id=str(i),
                     MIN='20',PTS='10',REB='4',AST='2') for i in range(1,6)]
        t=dict(target(),is_home=0,is_neutral_site=0)
        f=module.features_for_lineup_request(person,games(),t,['A','B','C','D','E'],
            confirmed=True,observed_at='2025-11-07T11:30:00+00:00',
            prediction_at='2025-11-07T11:35:00+00:00',scheduled_start_at='2025-11-07T12:00:00+00:00')
        self.assertTrue(f['live_prediction_ready'])
        self.assertTrue(f['eligible'])
        self.assertEqual(f['pts_mean_5'],10)


if __name__=='__main__':
    unittest.main()
