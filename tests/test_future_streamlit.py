from pathlib import Path
from datetime import date
from unittest.mock import patch
import unittest
import numpy as np
from streamlit.testing.v1 import AppTest
from tests.test_future_source import fixture,NOW
from tests.test_future_inference import history_fixture
from nba_dashboard.future_source import assess_context,SourceIssue

ROOT=Path(__file__).resolve().parents[1]


class Direct:
    def predict(self,x):return np.array([22.0]*len(x))


class Offset:
    def predict_components(self,rows):return {'points':np.array([28.0]*len(rows))}


class FuturePageTests(unittest.TestCase):
    def app(self):return AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run().switch_page('dashboard_pages/future.py').run()

    def test_page_waits_for_explicit_query(self):
        with patch('nba_dashboard.future_views.query_schedule',side_effect=AssertionError('unexpected network')):
            app=self.app()
            self.assertEqual(len(app.exception),0)
            self.assertIn('未來預測',[x.value for x in app.title])
            self.assertFalse(any(m.label=='預測得分' for m in app.metric))

    def test_future_schedule_is_not_reported_as_history_problem(self):
        bundle=history_fixture()
        bundle['pending']=[dict(nba_game_id='0022600174',GAMEDATE='2026-11-04',
            game_status='1',scheduled_start_utc='2026-11-05T00:00:00Z')]*2
        with patch('nba_dashboard.future_views.utc_now',return_value=NOW),patch('nba_dashboard.future_views.load_future',return_value=bundle):
            app=self.app()
            self.assertEqual(len(app.exception),0)
            self.assertFalse(any('狀態待核對' in x.value for x in app.warning))
            self.assertTrue(any('1 場尚未開賽' in x.value for x in app.caption))
            # The same scheduled game IS required history for a later target day.
            app.date_input(key='fp_day').set_value(date(2026,11,5)).run()
            self.assertTrue(any('1 場' in x.value and '停用預測' in x.value for x in app.warning))

    def test_prior_pending_blocks_and_in_progress_is_not_called_upcoming(self):
        bundle=history_fixture()
        bundle['pending']=[dict(nba_game_id='0022600174',GAMEDATE='2026-11-04',
            game_status='2',scheduled_start_utc='2026-11-04T22:00:00Z')]
        with patch('nba_dashboard.future_views.utc_now',return_value=NOW),patch('nba_dashboard.future_views.load_future',return_value=bundle):
            app=self.app()
            self.assertFalse(any('尚未開賽' in x.value for x in app.caption))
            self.assertTrue(any('1 場尚未收錄為完賽' in x.value for x in app.caption))
            app.date_input(key='fp_day').set_value(date(2026,11,5)).run()
            self.assertTrue(any('停用預測' in x.value for x in app.warning))

    def test_empty_schedule_and_source_error_show_no_prediction(self):
        with patch('nba_dashboard.future_views.utc_now',return_value=NOW),patch('nba_dashboard.future_views.query_schedule',return_value={'games':[],'source':{}}):
            app=self.app();app.button(key='fp_schedule_query').click().run()
            self.assertTrue(any('沒有NBA例行賽' in i.value for i in app.info))
            self.assertEqual(len(app.exception),0)
        with patch('nba_dashboard.future_views.query_schedule',side_effect=SourceIssue('尚未公布')):
            app=self.app();app.button(key='fp_schedule_query').click().run()
            self.assertTrue(any('尚未公布' in i.value for i in app.warning))

    def test_confirmed_fixture_can_predict_then_failed_refresh_clears_result(self):
        s,b,l=fixture();c=assess_context(s,b,l,NOW)
        models={'combined_prior25_w2':{'model':Direct()},'residual25_w2':Offset()}
        with patch('nba_dashboard.future_views.utc_now',return_value=NOW),patch('nba_dashboard.future_views.load_future',return_value=history_fixture()),patch('nba_dashboard.future_views.query_schedule',return_value={'games':[s],'source':{}}),patch('nba_dashboard.future_views.query_context',return_value=c) as source,patch('nba_dashboard.future_views.get_models',return_value=models):
            app=self.app();app.button(key='fp_schedule_query').click().run()
            app.button(key='fp_lineup_query').click().run()
            self.assertFalse(app.button(key='fp_predict').disabled)
            app.button(key='fp_predict').click().run()
            self.assertEqual(next(m.value for m in app.metric if m.label=='預測得分'),'22.00 分')
            self.assertEqual(len(app.exception),0)
            self.assertGreaterEqual(source.call_count,2)
            self.assertIsNotNone(app.session_state['fp_result'])
            source.side_effect=SourceIssue('尚未公布')
            app.button(key='fp_lineup_query').click().run()
            self.assertFalse(any(m.label=='預測得分' for m in app.metric))
            self.assertEqual(len(app.exception),0)

    def test_projected_never_enables_prediction(self):
        s,b,l=fixture();l['data']['games'][0]['homeTeam']['players'][0]['lineupStatus']='Projected';c=assess_context(s,b,l,NOW)
        with patch('nba_dashboard.future_views.utc_now',return_value=NOW),patch('nba_dashboard.future_views.load_future',return_value=history_fixture()),patch('nba_dashboard.future_views.query_schedule',return_value={'games':[s],'source':{}}),patch('nba_dashboard.future_views.query_context',return_value=c):
            app=self.app();app.button(key='fp_schedule_query').click().run();app.button(key='fp_lineup_query').click().run()
            self.assertTrue(app.button(key='fp_predict').disabled)
            self.assertFalse(any(m.label=='預測得分' for m in app.metric))


if __name__=='__main__':unittest.main()
