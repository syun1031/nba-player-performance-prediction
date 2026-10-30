from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


class ResearchStreamlitTests(unittest.TestCase):
    def app(self):return AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run().switch_page('dashboard_pages/comparison.py').run()
    def metric(self,app,label):return next(m.value for m in app.metric if m.label==label)

    def test_2024_test_and_high_scores_show_three_models_on_same_cohort(self):
        app=self.app();app.selectbox(key='comparison_season').set_value('2024-25').run()
        self.assertEqual(len(app.exception),0);self.assertEqual(self.metric(app,'樣本數'),'7,060')
        self.assertEqual(self.metric(app,'原九項＋上季表現 MAE'),'4.743 分')
        self.assertEqual(self.metric(app,'增減模型＋上季表現＋高分2倍 MAE'),'4.810 分')
        self.assertEqual(len(app.dataframe[0].value),5)
        app.selectbox(key='research_threshold').set_value(30).run()
        self.assertEqual(self.metric(app,'樣本數'),'277')
        self.assertEqual(self.metric(app,'1＋3＋同區＋上季表現＋2倍 MAE'),'11.600 分')
        self.assertEqual(self.metric(app,'增減模型＋上季表現＋高分2倍 MAE'),'11.578 分')
        columns=app.dataframe[1].value.columns.tolist()
        self.assertLess(columns.index('實際上場時間（分鐘）'),columns.index('實際得分'))
        self.assertIn('增減模型＋上季表現＋高分2倍',columns)
        self.assertTrue(any('賽後' in x.value for x in app.info))
        self.assertGreaterEqual(len(app.get('download_button')),2)

    def test_season_period_switch_keeps_counts_and_marks_reused_test(self):
        app=self.app();app.radio(key='comparison_scope').set_value('research').run()
        self.assertEqual(self.metric(app,'樣本數'),'6,979')
        self.assertEqual(self.metric(app,'增減模型＋上季表現＋高分2倍 MAE'),'4.803 分')
        app.radio(key='research_period').set_value('CV').run()
        self.assertEqual(self.metric(app,'樣本數'),'9,023')
        self.assertTrue(any('2025-11-30' in x.value and '2026-01-15' in x.value for x in app.caption))
        app.selectbox(key='comparison_season').set_value('2024-25').run()
        self.assertEqual(self.metric(app,'樣本數'),'8,647')
        self.assertTrue(any('2024-11-30' in x.value and '2025-01-15' in x.value for x in app.caption))
        app.radio(key='research_period').set_value('test').run()
        app.selectbox(key='comparison_season').set_value('2025-26').run()
        self.assertEqual(self.metric(app,'樣本數'),'6,979')
        self.assertTrue(any('追加模型比較' in x.value for x in app.info))
        app.radio(key='comparison_scope').set_value('classic').run()
        self.assertEqual(self.metric(app,'主模型 MAE'),'4.747 分');self.assertEqual(len(app.exception),0)


if __name__=='__main__':unittest.main()
