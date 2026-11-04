import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from tests.test_future_inference import history_fixture

ROOT=Path(__file__).resolve().parents[1]


class CurrentPageTests(unittest.TestCase):
    def app(self):
        app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()
        return app.selectbox(key='history_season').set_value('2026-27').run()

    def test_current_season_minutes_chart_and_download(self):
        # Stable nonempty fixture: a legitimate empty production package must
        # not cause the daily publication job to fail its UI checks.
        row=dict(nba_player_id='100',player_id='WAS 0',game_id='WAS VS. ATL|2026-11-04',GAMEDATE='2026-11-04',
            pred_route25=22.,pred_baseline=20.,pred_combined_prior25_w2=22.,pred_residual25_w2=23.,
            route25_model='combined_prior25_w2',actual_minutes=30.,target_pts=24.)
        data={'frame':pd.DataFrame([row]),'manifest':{'collected_at_utc':'2026-11-04T09:00:00Z','last_completed_date':'2026-11-04','eligible_rows':1,'excluded_rows':5}}
        with patch('nba_dashboard.current_views.load_current',return_value=data),patch('nba_dashboard.current_views.load_future',return_value=history_fixture()):
            app=self.app()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('假設性回測' in x.value for x in app.caption))
        labels=[x.label for x in app.metric]
        self.assertEqual(labels.index('實際上場時間')+1,labels.index('實際得分'))
        self.assertTrue(len(app.dataframe)>=2)
        self.assertTrue(any('本季' in x.value for x in app.subheader))

    def test_empty_current_dataset_shows_message(self):
        with patch('nba_dashboard.current_views.load_current',return_value={'frame':pd.DataFrame(),'manifest':{'collected_at_utc':'2026-11-03T06:39:00Z','last_completed_date':None,'eligible_rows':0,'excluded_rows':0}}):
            app=self.app()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('尚無' in x.value for x in app.info))

    def test_switch_back_preserves_old_season(self):
        app=self.app()
        app.selectbox(key='history_season').set_value('2025-26').run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('固定顯示3月至季末' in x.value for x in app.caption))
