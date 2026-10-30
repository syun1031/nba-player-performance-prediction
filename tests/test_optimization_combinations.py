import unittest
from unittest.mock import patch
import numpy as np

from scripts.combination_models import CASES,CombinationTwoStage,rank_cases
from scripts.experiment_recent_changes import BASE
from scripts.recent_change_features import NEW_FEATURES
from scripts.opponent_context import CONTEXT_FEATURES
from scripts.time_validation_tuning import FitBudget
from tests.test_two_stage_model import RecordingModel


class CombinationTests(unittest.TestCase):
    def row(self):
        return dict.fromkeys(BASE+NEW_FEATURES+CONTEXT_FEATURES,1.)|dict(
            player_id='P',GAMEDATE='2025-11-01',split='train',target_pts=12,target_minutes=24)

    def test_four_cases_have_exact_independent_feature_sets_and_model_modes(self):
        self.assertEqual(list(CASES),['case12','case13','case23','case123'])
        self.assertEqual([len(v['columns']) for v in CASES.values()],[14,16,11,16])
        self.assertEqual([v['mode'] for v in CASES.values()],['two_stage','direct','two_stage','two_stage'])
        self.assertTrue(set(NEW_FEATURES).isdisjoint(CASES['case23']['columns']))
        self.assertTrue(set(CONTEXT_FEATURES).isdisjoint(CASES['case12']['columns']))
        self.assertEqual(CASES['case13']['columns'],CASES['case123']['columns'])

    def test_case23_ignores_recent_change_features_and_outcome_labels_at_prediction(self):
        model=CombinationTwoStage({},'case23'); model.minutes_model=RecordingModel(30); model.rate_model=RecordingModel(.5)
        row=self.row(); row.update(dict.fromkeys(NEW_FEATURES,float('nan')))
        row.update(target_pts=999,target_minutes=999,target_ppm=999)
        self.assertEqual(model.predict_components([row])['points'].tolist(),[15.])
        no_labels={k:v for k,v in row.items() if not k.startswith('target_')}
        np.testing.assert_equal(model.predict_components([no_labels])['points'],[15])

    def test_case12_does_not_require_or_use_opponent_features(self):
        model=CombinationTwoStage({},'case12'); model.minutes_model=RecordingModel(20); model.rate_model=RecordingModel(1)
        row={k:v for k,v in self.row().items() if k not in CONTEXT_FEATURES}
        np.testing.assert_equal(model.predict_components([row])['points'],[20])

    def test_case123_requires_both_recent_and_opponent_features(self):
        model=CombinationTwoStage({},'case123'); model.minutes_model=RecordingModel(20); model.rate_model=RecordingModel(1)
        row=self.row(); del row[CONTEXT_FEATURES[0]]
        with self.assertRaises((ValueError,KeyError)):
            model.predict_components([row])

    def test_column_configuration_does_not_mutate_older_model_specification(self):
        from scripts.two_stage_model import TwoStageModel,FEATURES
        CombinationTwoStage({},'case23')
        self.assertEqual(TwoStageModel({}).columns,BASE+NEW_FEATURES)
        self.assertEqual(FEATURES,BASE+NEW_FEATURES)

    def test_training_uses_selected_columns_and_keeps_weighted_rate_contract(self):
        a,b=RecordingModel(),RecordingModel()
        budget=FitBudget(2)
        with patch('scripts.two_stage_model.XGBRegressor',side_effect=[a,b]):
            model=CombinationTwoStage({},'case23').fit([self.row()],budget)
        self.assertEqual(a.x.shape,(1,11)); self.assertEqual(b.x.shape,(1,11))
        np.testing.assert_equal(b.y,[.5]); np.testing.assert_equal(b.weight,[1])
        self.assertEqual(budget.count,2); self.assertEqual(model.case,'case23')

    def test_unknown_and_direct_cases_cannot_be_constructed_as_two_stage(self):
        for case in ['unknown','case13']:
            with self.assertRaises(ValueError): CombinationTwoStage({},case)

    def test_ranking_uses_mae_then_rmse_then_complexity_not_input_order(self):
        rows=[dict(variant=c,mae=4.,rmse=6.) for c in reversed(CASES)]
        ordered=rank_cases(rows)
        self.assertEqual([r['variant'] for r in ordered],['case23','case12','case13','case123'])
        rows[0].update(mae=3.99,rmse=9.)
        self.assertEqual(rank_cases(rows)[0]['variant'],'case123')
        rows[0].update(mae=4.,rmse=5.)
        self.assertEqual(rank_cases(rows)[0]['variant'],'case123')

    def test_ranking_rejects_missing_duplicate_or_nonfinite_cases(self):
        rows=[dict(variant=c,mae=4.,rmse=6.) for c in CASES]
        for bad in [rows[:3],rows+[rows[0]],[dict(r,mae=float('nan')) for r in rows]]:
            with self.assertRaises(ValueError): rank_cases(bad)


if __name__=='__main__':
    unittest.main()
