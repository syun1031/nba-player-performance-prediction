from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


class JournalInterfaceTests(unittest.TestCase):
    def test_shared_sidebar_navigation_and_brand_across_three_pages(self):
        app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()
        for page in ['dashboard_pages/history.py','dashboard_pages/comparison.py','dashboard_pages/future.py']:
            app.switch_page(page).run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(app.sidebar.get('page_link')),3)
            self.assertTrue(any('court' in x.value and 'notes' in x.value for x in app.sidebar.markdown))
            self.assertFalse(len(app.main.get('page_link')))

    def test_history_player_report_preserves_metrics(self):
        app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()
        selected=next(x.value for x in app.selectbox if x.label=='球員')
        self.assertEqual(app.title[0].value,selected)
        labels=[x.label for x in app.metric]
        self.assertEqual(labels.index('實際上場時間')+1,labels.index('實際得分'))
