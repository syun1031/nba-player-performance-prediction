import unittest
from pathlib import Path
import tempfile
import shutil
import json
import hashlib

from nba_dashboard.data import (
    DataIssue, validate_predictions, recent_history, comparison_metrics, load_bundle, prediction_table, expected_hash, FILES, file_revision,
)
import pandas as pd


class DashboardDataTests(unittest.TestCase):
    def predictions(self):
        return pd.DataFrame([
            dict(player_id='P',game_id='AAA VS. BBB|2026-03-01',GAMEDATE='2026-03-01',target_pts=30.,pred_baseline=20.,pred_xgboost_starter=25.),
            dict(player_id='Q',game_id='BBB @ AAA|2026-03-01',GAMEDATE='2026-03-01',target_pts=10.,pred_baseline=8.,pred_xgboost_starter=14.)])

    def test_recent_history_excludes_zero_same_day_future_and_other_player(self):
        history=pd.DataFrame([dict(PLAYER=p,GAMEDATE=d,MIN=m,PTS=s) for p,d,m,s in [
            ('P','2026-02-25',20,12),('P','2026-02-26',0,0),('P','2026-03-01',25,30),
            ('P','2026-03-02',25,40),('Q','2026-02-27',25,20)]])
        result=recent_history(history,'P','2026-03-01')
        self.assertEqual(result['PTS'].tolist(),[12])

    def test_recent_history_uses_ten_latest_without_mutating_source(self):
        history=pd.DataFrame([dict(PLAYER='P',GAMEDATE=f'2026-02-{i:02d}',MIN=20,PTS=i) for i in range(1,15)])
        result=recent_history(history,'P','2026-03-01')
        self.assertEqual(result['PTS'].tolist(),list(range(5,15)))
        self.assertEqual(len(history),14)

    def test_predictions_reject_duplicate_nonfinite_and_out_of_split_dates(self):
        good=self.predictions()
        validate_predictions(good,expected_n=2,start='2026-03-01',end='2026-04-12',models=['xgboost_starter'])
        bads=[pd.concat([good,good.iloc[:1]],ignore_index=True),good.assign(target_pts=float('nan')),
              good.assign(GAMEDATE='2026-02-28'),
              good.assign(GAMEDATE='2026-03-32',game_id=good['game_id'].str.replace('2026-03-01','2026-03-32',regex=False))]
        for bad in bads:
            with self.assertRaises(DataIssue):
                validate_predictions(bad,expected_n=2,start='2026-03-01',end='2026-04-12',models=['xgboost_starter'])

    def test_threshold_comparison_keeps_baseline_and_model_on_identical_samples(self):
        result=comparison_metrics(self.predictions(),['baseline','xgboost_starter'],threshold=30)
        self.assertEqual(result['n'].tolist(),[1,1])
        self.assertEqual(result['mae'].tolist(),[10.,5.])
        self.assertEqual(result['rmse'].tolist(),[10.,5.])

    def test_empty_subgroup_has_no_fabricated_zero_score(self):
        result=comparison_metrics(self.predictions(),['baseline','xgboost_starter'],threshold=60)
        self.assertTrue(result.empty)

    def test_prediction_table_uses_signed_description_and_human_headers(self):
        table=prediction_table(self.predictions(),'xgboost_starter')
        self.assertEqual(table['預測狀況'].tolist(),['低估 5.00 分','高估 4.00 分'])
        self.assertEqual(table['絕對誤差'].tolist(),[5.,4.])

    def test_missing_data_is_reported_as_a_clear_data_issue(self):
        with self.assertRaises(DataIssue):
            load_bundle(Path('tests/nonexistent_dashboard_root'))

    def test_real_frozen_bundle_reconciles_test_and_validation_scores(self):
        bundle=load_bundle(Path(__file__).resolve().parents[1])
        self.assertEqual(bundle.primary,'xgboost_starter')
        self.assertEqual(len(bundle.predictions['test']),6979)
        self.assertEqual(len(bundle.predictions['validation']),3462)
        result=comparison_metrics(bundle.predictions['test'],['xgboost_starter'])
        self.assertAlmostEqual(result.iloc[0]['mae'],4.7475,places=4)

    def test_windows_evidence_paths_are_readable_on_other_platforms(self):
        self.assertEqual(expected_hash({'outputs\\data\\scores.csv':'abc'},Path('outputs/data/scores.csv')),'abc')
        with self.assertRaises(DataIssue):
            expected_hash({'outputs\\data.csv':'abc','outputs/data.csv':'different'},Path('outputs/data.csv'))

    def test_self_consistent_validation_replacement_is_rejected_and_invalidates_cache(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            copied=Path(directory)
            for relative in FILES.values():
                target=copied/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(root/relative,target)
            revision=file_revision(copied)
            prediction=copied/FILES['validation_predictions']
            prediction.write_bytes(prediction.read_bytes()+b'\n')
            evidence=copied/FILES['validation_evidence']
            meta=json.loads(evidence.read_text(encoding='utf-8'))
            meta['output_sha256']['predictions_february.csv']=hashlib.sha256(prediction.read_bytes()).hexdigest()
            evidence.write_text(json.dumps(meta),encoding='utf-8')
            self.assertNotEqual(revision,file_revision(copied))
            with self.assertRaises(DataIssue):
                load_bundle(copied)


if __name__=='__main__':
    unittest.main()
