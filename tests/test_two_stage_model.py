import unittest
from unittest.mock import patch
import numpy as np

from scripts.two_stage_model import TwoStageModel,attach_minute_targets,FEATURES
from scripts.time_validation_tuning import FitBudget


def sample(day='2025-11-01',points=0,minutes=12):
    return dict.fromkeys(FEATURES,1.)|dict(player_id='P',game_id='AAA @ BBB|'+day,GAMEDATE=day,
        split='train',target_pts=points,target_minutes=minutes)


class RecordingModel:
    def __init__(self,prediction=1.):
        self.prediction=prediction

    def fit(self,x,y,**kwargs):
        self.x=x.copy(); self.y=y.copy(); self.weight=kwargs.get('sample_weight')
        return self

    def predict(self,x):
        return np.full(len(x),self.prediction)


class TwoStageModelTests(unittest.TestCase):
    def test_targets_join_exact_player_and_game_without_mutating_inputs(self):
        row=sample(); del row['target_minutes']
        history=[dict(player_id='P',game_id=row['game_id'],GAMEDATE=row['GAMEDATE'],MIN='12.5',PTS='0')]
        result=attach_minute_targets([row],history)
        self.assertEqual(result[0]['target_minutes'],12.5)
        self.assertEqual(result[0]['target_ppm'],0)
        self.assertNotIn('target_minutes',row)
        self.assertTrue(all(result[0][key]==value for key,value in row.items()))

    def test_target_join_rejects_missing_duplicate_mismatch_and_test_dates(self):
        row=sample(points=10)
        h=dict(player_id='P',game_id=row['game_id'],GAMEDATE=row['GAMEDATE'],MIN='12',PTS='10')
        for history in [[],[h,h],[dict(h,PTS='11')],[dict(h,MIN='0')],[dict(h,MIN='nan')]]:
            with self.assertRaises(ValueError):
                attach_minute_targets([row],history)
        with self.assertRaises(ValueError):
            attach_minute_targets([dict(row,GAMEDATE='2026-03-01',split='test')],[h])
        with self.assertRaises(ValueError):
            attach_minute_targets([row,row],[h])

    def test_fit_uses_training_minutes_as_normalized_rate_weights(self):
        a,b=RecordingModel(),RecordingModel()
        rows=[sample(points=0,minutes=10),sample('2025-11-02',points=30,minutes=30)]
        budget=FitBudget(2)
        with patch('scripts.two_stage_model.XGBRegressor',side_effect=[a,b]):
            model=TwoStageModel({}).fit(rows,budget)
        np.testing.assert_equal(a.y,[10,30]); np.testing.assert_equal(b.y,[0,1])
        np.testing.assert_equal(b.weight,[.5,1.5])
        self.assertIsNone(a.weight)
        self.assertEqual(a.x.shape,(2,14)); self.assertEqual(budget.count,2)
        self.assertEqual(model.trained_through,'2025-11-02')

    def test_prediction_is_product_and_never_uses_target_outcomes(self):
        model=TwoStageModel({}); model.minutes_model=RecordingModel(32); model.rate_model=RecordingModel(.75)
        a=model.predict_components([sample()])
        b=model.predict_components([dict(sample(),target_pts=999,target_minutes=999,target_ppm=999)])
        np.testing.assert_equal(a['points'],[24]); np.testing.assert_equal(a['points'],b['points'])
        no_label={k:v for k,v in sample().items() if not k.startswith('target_')}
        np.testing.assert_equal(model.predict_components([no_label])['points'],[24])

    def test_negative_components_clip_to_zero_and_overtime_is_not_capped(self):
        model=TwoStageModel({}); model.minutes_model=RecordingModel(-1); model.rate_model=RecordingModel(-2)
        a=model.predict_components([sample()]); np.testing.assert_equal(a['points'],[0])
        model.minutes_model=RecordingModel(53); model.rate_model=RecordingModel(1)
        np.testing.assert_equal(model.predict_components([sample()])['points'],[53])

    def test_invalid_labels_or_features_do_not_trigger_a_fit(self):
        for row in [dict(sample(),target_minutes=0),dict(sample(),target_pts=-1),
                    dict(sample(),target_minutes=float('nan')),dict(sample(),pts_mean_5=float('inf')),
                    dict(sample(),GAMEDATE='2026-02-01')]:
            budget=FitBudget(2)
            with self.assertRaises(ValueError):
                TwoStageModel({}).fit([row],budget)
            self.assertEqual(budget.count,0)

    def test_nonfinite_model_predictions_are_rejected(self):
        model=TwoStageModel({}); model.minutes_model=RecordingModel(float('nan')); model.rate_model=RecordingModel(1)
        with self.assertRaises(ValueError):
            model.predict_components([sample()])


if __name__=='__main__':
    unittest.main()
