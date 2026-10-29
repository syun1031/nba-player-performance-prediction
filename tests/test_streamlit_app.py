import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from nba_dashboard.data import load_bundle

ROOT=Path(__file__).resolve().parents[1]


class StreamlitPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=load_bundle(ROOT)

    def app(self):
        return AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()

    def metric(self,app,label):
        return next(m.value for m in app.metric if m.label==label)

    def test_history_switches_player_and_game_without_stale_prediction(self):
        app=self.app()
        self.assertEqual(len(app.exception),0)
        for player in ['Luka Dončić','Jarrett Allen']:
            app.selectbox(key='history_player').set_value(player).run()
            game=app.selectbox(key='history_game').value
            data=self.bundle.predictions['test']
            expected=data.loc[(data['player_id']==player)&(data['game_id']==game)].iloc[0]
            self.assertEqual(self.metric(app,'模型預測'),f'{expected["pred_xgboost_starter"]:.2f} 分')
            other=data.loc[data['player_id']==player].iloc[0]['game_id']
            app.selectbox(key='history_game').set_value(other).run()
            expected=data.loc[(data['player_id']==player)&(data['game_id']==other)].iloc[0]
            self.assertEqual(self.metric(app,'實際得分'),f'{expected["target_pts"]:.0f} 分')
            raw=self.bundle.history
            matchup,day=other.rsplit('|',1)
            minutes=raw.loc[(raw['PLAYER']==player)&(raw['GAMEDATE']==day)&(raw['MATCHUP']==matchup),'MIN'].iloc[0]
            self.assertEqual(self.metric(app,'實際上場時間'),f'{minutes:.2f} 分鐘')
            labels=[m.label for m in app.metric]
            self.assertEqual(labels.index('實際上場時間')+1,labels.index('實際得分'))
            table=app.dataframe[0].value
            selected=table.loc[(table['比賽日期']==day)&(table['對戰']==matchup)].iloc[0]
            self.assertEqual(selected['實際上場時間（分鐘）'],minutes)
            columns=table.columns.tolist()
            self.assertEqual(columns.index('實際上場時間（分鐘）')+1,columns.index('實際得分'))
            self.assertEqual(len(app.exception),0)

    def test_navigation_and_dataset_high_score_switch_reconciles_counts(self):
        app=self.app().switch_page('dashboard_pages/comparison.py').run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(self.metric(app,'樣本數'),'6,979')
        app.selectbox(key='comparison_threshold').set_value(30).run()
        self.assertEqual(self.metric(app,'樣本數'),'228')
        app.radio(key='comparison_period').set_value('validation').run()
        self.assertEqual(self.metric(app,'樣本數'),'99')
        self.assertEqual(len(app.exception),0)

    def test_feature_version_changes_chart_not_locked_primary(self):
        app=self.app().switch_page('dashboard_pages/comparison.py').run()
        app.selectbox(key='comparison_features').set_value('teammates').run()
        self.assertEqual(self.metric(app,'主模型 MAE'),'4.747 分')
        table=app.dataframe[0].value
        self.assertEqual(len(table),4)
        self.assertEqual(int(table['已鎖定'].sum()),0)
        self.assertEqual(len(app.exception),0)

    def test_empty_model_selection_keeps_baseline_and_explains_it(self):
        app=self.app().switch_page('dashboard_pages/comparison.py').run()
        app.multiselect(key='comparison_families').set_value([]).run()
        self.assertEqual(len(app.dataframe[0].value),1)
        self.assertTrue(any('只顯示五場均值' in item.value for item in app.info))
        self.assertEqual(len(app.exception),0)

    def test_missing_data_shows_a_clear_error_without_traceback(self):
        with patch('nba_dashboard.views.ROOT',ROOT/'tests/not_a_data_directory'):
            app=self.app()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('資料' in e.value for e in app.error))
            self.assertEqual(len(app.metric),0)

    def test_downloads_and_research_limitations_are_visible(self):
        app=self.app()
        self.assertGreaterEqual(len(app.get('download_button')),1)
        self.assertTrue(any('30分以上' in x.value for x in app.warning))
        self.assertTrue(any('賽後快照' in x.value for x in app.caption))


if __name__=='__main__':
    unittest.main()
