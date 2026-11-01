import unittest,tempfile,shutil,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from nba_dashboard.routing_data import apply_route,load_routing,recent_games,ROUTE,comparison_metrics
from nba_dashboard.data import DataIssue
ROOT=Path(__file__).resolve().parents[1]

class RoutingDataTests(unittest.TestCase):
    def test_boundary_and_outcomes_cannot_choose_branch(self):
        f=pd.DataFrame(dict(pred_baseline=[24.999,25.,30.],pred_combined_prior25_w2=[10.,11.,12.],pred_residual25_w2=[20.,21.,22.]))
        a=apply_route(f);self.assertEqual(a['pred_'+ROUTE].tolist(),[10,21,22])
        self.assertEqual(a.route25_model.tolist(),['combined_prior25_w2','residual25_w2','residual25_w2'])
        b=apply_route(f.assign(target_pts=[99,0,0],actual_minutes=[0,99,99]))
        np.testing.assert_equal(a['pred_'+ROUTE].to_numpy(),b['pred_'+ROUTE].to_numpy())
        with self.assertRaises(DataIssue):apply_route(f.assign(pred_baseline=[np.nan,25,30]))

    def test_two_seasons_all_periods_same_cohort_and_strict_history(self):
        bundle=load_routing(ROOT)
        for season,periods in bundle.predictions.items():
            for period,f in periods.items():
                expected=np.where(f.pred_baseline>=25,f.pred_residual25_w2,f.pred_combined_prior25_w2)
                np.testing.assert_equal(f['pred_'+ROUTE].to_numpy(),expected)
                self.assertEqual(len(comparison_metrics(f)),6)
        f=bundle.predictions['2024-25']['test'];r=f[f.player_id.eq('Luka Dončić')].iloc[0]
        history=recent_games(bundle.histories['2024-25'],'Luka Dončić',r.GAMEDATE)
        self.assertLessEqual(len(history),10);self.assertTrue((history.GAMEDATE<r.GAMEDATE).all());self.assertTrue((history.MIN>0).all())
        self.assertAlmostEqual(history.tail(5).PTS.mean(),r.pred_baseline)

    def test_missing_or_tampered_package_stops(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'outputs/dashboard_routing';shutil.copytree(ROOT/'outputs/dashboard_routing',out)
            shutil.copytree(ROOT/'outputs/dashboard_comparison',Path(d)/'outputs/dashboard_comparison')
            p=out/'2024_25_test.csv';p.write_bytes(p.read_bytes()+b'\n')
            with self.assertRaises(DataIssue):load_routing(Path(d))
            p.unlink()
            with self.assertRaises(DataIssue):load_routing(Path(d))
