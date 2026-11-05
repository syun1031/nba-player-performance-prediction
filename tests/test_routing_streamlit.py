import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from nba_dashboard.routing_data import load_routing
ROOT=Path(__file__).resolve().parents[1]

class RoutingPageTests(unittest.TestCase):
    def app(self):
        app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()
        return app.selectbox(key='history_season').set_value('2025-26').run()
    def metric(self,app,label):return next(m.value for m in app.metric if m.label==label)
    def test_history_both_seasons_minutes_and_both_branches(self):
        bundle=load_routing(ROOT);app=self.app()
        for season in ['2025-26','2024-25']:
            app.selectbox(key='history_season').set_value(season).run()
            f=bundle.predictions[season]['test']
            for branch in ['combined_prior25_w2','residual25_w2']:
                r=f.loc[f.route25_model.eq(branch)].iloc[0]
                next(x for x in app.selectbox if x.label=='球員').set_value(r.player_id).run()
                next(x for x in app.selectbox if x.label=='比賽（美東日期）').set_value(r.game_id).run()
                self.assertEqual(len(app.exception),0)
                self.assertEqual(self.metric(app,'模型預測'),f'{r.pred_route25:.2f} 分')
                self.assertTrue(any(('直接25項' if branch=='combined_prior25_w2' else '五場均值增減25項') in t.value for t in app.info))
                labels=[m.label for m in app.metric];self.assertEqual(labels.index('實際上場時間')+1,labels.index('實際得分'))
                recent=app.dataframe[0].value;self.assertTrue((recent['比賽日期（美東）']<r.GAMEDATE).all());self.assertLessEqual(len(recent),10)
                columns=recent.columns.tolist();self.assertEqual(columns.index('實際上場時間（分鐘）')+1,columns.index('實際得分'))
                table=app.dataframe[1].value;self.assertIn('本場選用模型',table);self.assertIn('模型預測',table)
    def test_comparison_has_route_and_originals_all_periods(self):
        app=self.app().switch_page('dashboard_pages/comparison.py').run()
        for season,count in [('2025-26','6,979'),('2024-25','7,060')]:
            app.selectbox(key='comparison_season').set_value(season).run()
            self.assertEqual(self.metric(app,'樣本數'),count);self.assertEqual(len(app.dataframe[0].value),6)
            self.assertTrue(any('25分切換方案' in m.label for m in app.metric))
            app.selectbox(key='research_threshold').set_value(30).run();self.assertEqual(len(app.exception),0)
            app.selectbox(key='research_threshold').set_value(None).run()
        app.radio(key='research_period').set_value('CV').run();self.assertEqual(self.metric(app,'樣本數'),'8,647')
    def test_game_widget_identity_changes_with_player_context(self):
        app=self.app()
        player_box=next(x for x in app.selectbox if x.label=='球員')
        game_box=next(x for x in app.selectbox if x.label=='比賽（美東日期）')
        first_id=game_box.proto.id
        player_box.set_value('Luka Dončić').run()
        game_box=next(x for x in app.selectbox if x.label=='比賽（美東日期）')
        self.assertNotEqual(game_box.proto.id,first_id)
        self.assertEqual(game_box.value,'LAL @ OKC|2026-04-02')
        before_player=next(x for x in app.selectbox if x.label=='球員').proto.id
        app.selectbox(key='history_season').set_value('2024-25').run()
        self.assertNotEqual(next(x for x in app.selectbox if x.label=='球員').proto.id,before_player)
        self.assertEqual(next(x for x in app.selectbox if x.label=='比賽（美東日期）').value,'LAL VS. HOU|2025-04-11')
    def test_history_only_march_onward_without_period_selector(self):
        app=self.app()
        self.assertFalse(any(r.key=='history_period' for r in app.radio))
        app.session_state['history_period']='validation'
        for season,year in [('2025-26','2026'),('2024-25','2025')]:
            app.selectbox(key='history_season').set_value(season).run()
            options=next(x for x in app.selectbox if x.label=='比賽（美東日期）').options
            self.assertTrue(all(str(day).startswith(year+'-03') or str(day).startswith(year+'-04') for day in options))
            table=app.dataframe[1].value
            self.assertTrue(table['比賽日期（美東）'].ge(year+'-03-01').all())
            self.assertEqual(len(app.exception),0)
        app.switch_page('dashboard_pages/comparison.py').run()
        self.assertTrue(any('CV' in str(x) for x in app.radio(key='research_period').options))
    def test_history_missing_package_shows_clear_error(self):
        from unittest.mock import patch
        with patch('nba_dashboard.routing_views.ROOT',ROOT/'tests/missing-routing-data'):
            app=self.app();self.assertEqual(len(app.exception),0);self.assertTrue(len(app.error)>0)
