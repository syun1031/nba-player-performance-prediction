import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
from nba_dashboard.research_data import load_research, research_metrics,DataIssue,MODELS

ROOT=Path(__file__).resolve().parents[1]


class ResearchDashboardTests(unittest.TestCase):
    def test_two_seasons_three_models_and_saved_metrics(self):
        bundle=load_research(ROOT)
        expected={'2025-26':{'CV':9023,'validation':3462,'test':6979},'2024-25':{'CV':8647,'validation':3634,'test':7060}}
        for season,periods in expected.items():
            for period,n in periods.items():
                f=bundle.predictions[season][period]
                self.assertEqual(len(f),n)
                self.assertEqual(research_metrics(f,['baseline']+MODELS)['n'].tolist(),[n]*5)
        self.assertAlmostEqual(research_metrics(bundle.predictions['2024-25']['test'],['basic_prior14']).iloc[0]['mae'],4.743392476312837)
        self.assertEqual(len(bundle.predictions['2025-26']['test']),6979)

    def test_tampered_and_missing_bundle_stop(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'outputs/dashboard_comparison';shutil.copytree(ROOT/'outputs/dashboard_comparison',target)
            f=target/'2024_25_test.csv';f.write_bytes(f.read_bytes()+b'\n')
            with self.assertRaises(DataIssue):load_research(Path(d))
            f.unlink()
            with self.assertRaises(DataIssue):load_research(Path(d))

    def test_high_score_same_sample_and_unknown_model_rejected(self):
        f=pd.DataFrame([dict(target_pts=30,pred_baseline=20,pred_basic9=25),dict(target_pts=0,pred_baseline=5,pred_basic9=2)])
        m=research_metrics(f,['baseline','basic9'],30)
        self.assertEqual(m['n'].tolist(),[1,1]);self.assertEqual(m['mae'].tolist(),[10,5])
        with self.assertRaises(DataIssue):research_metrics(f,['not_a_model'])


if __name__=='__main__':unittest.main()
