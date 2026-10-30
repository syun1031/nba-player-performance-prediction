import unittest
from pathlib import Path
from nba_dashboard.research_data import load_research,MODELS,research_metrics

ROOT=Path(__file__).resolve().parents[1]
class FourModelTests(unittest.TestCase):
    def test_both_seasons_have_offset_on_identical_cohort(self):
        self.assertIn('residual25_w2',MODELS)
        bundle=load_research(ROOT)
        for season,periods in bundle.predictions.items():
            for period,frame in periods.items():
                self.assertEqual(len(research_metrics(frame,['baseline']+MODELS)),5)
                self.assertFalse(frame['pred_residual25_w2'].isna().any())
                self.assertTrue((frame['pred_residual25_w2']>=0).all())
        self.assertAlmostEqual(research_metrics(bundle.predictions['2025-26']['test'],['residual25_w2']).iloc[0]['mae'],4.8030026435)
