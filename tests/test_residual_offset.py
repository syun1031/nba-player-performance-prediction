import unittest
from unittest.mock import patch
import numpy as np
from scripts.residual_offset import ResidualOffsetModel,COLUMNS
from scripts.time_validation_tuning import FitBudget
from tests.test_two_stage_model import RecordingModel


class ResidualOffsetTests(unittest.TestCase):
    def row(self,day='2025-11-01',actual=10,baseline=20):
        return dict.fromkeys(COLUMNS,1.)|dict(player_id='P',game_id='LAL @ DEN|'+day,GAMEDATE=day,
            split='train',target_pts=actual,pts_mean_5=baseline)

    def test_negative_delta_labels_and_weights_use_original_points(self):
        base=RecordingModel(); rows=[self.row(),self.row('2025-11-02',40,30)]
        with patch('scripts.residual_offset.XGBRegressor',return_value=base):
            model=ResidualOffsetModel({},2).fit(rows,FitBudget(1))
        np.testing.assert_equal(base.y,[-10,10]); np.testing.assert_equal(base.weight,[2/3,4/3])
        self.assertEqual(model.weight_normalizer,1.5)

    def test_unweighted_fit_has_constant_one_weights(self):
        base=RecordingModel()
        with patch('scripts.residual_offset.XGBRegressor',return_value=base):
            ResidualOffsetModel({},1).fit([self.row(),self.row('2025-11-02',50,30)],FitBudget(1))
        np.testing.assert_equal(base.weight,[1,1])

    def test_adds_delta_back_to_each_players_baseline(self):
        model=ResidualOffsetModel({},1); model.model=RecordingModel(-3)
        result=model.predict_components([self.row(baseline=20),self.row(baseline=40)])
        np.testing.assert_equal(result['delta'],[-3,-3]); np.testing.assert_equal(result['points'],[17,37])

    def test_only_final_score_is_clipped_no_upper_cap(self):
        model=ResidualOffsetModel({},2); model.model=RecordingModel(-25)
        r=model.predict_components([self.row(baseline=20)])
        np.testing.assert_equal(r['delta'],[-25]); np.testing.assert_equal(r['raw_points'],[-5]); np.testing.assert_equal(r['points'],[0])
        model.model=RecordingModel(25)
        np.testing.assert_equal(model.predict_components([self.row(baseline=40)])['points'],[65])

    def test_current_outcomes_are_ignored_and_absent_at_inference(self):
        model=ResidualOffsetModel({},1); model.model=RecordingModel(2)
        row=self.row(); expected=model.predict_components([row])['points']
        changed=dict(row,target_pts=99999,PTS=99999,MIN=99999,target_delta=99999)
        np.testing.assert_equal(model.predict_components([changed])['points'],expected)
        del row['target_pts']; np.testing.assert_equal(model.predict_components([row])['points'],expected)

    def test_missing_head_to_head_keeps_native_nan(self):
        model=ResidualOffsetModel({},1); model.model=RecordingModel(0)
        row=self.row(); row.update(same_conf_h2h_count=0,same_conf_h2h_pts_last_1='',same_conf_h2h_pts_mean='')
        np.testing.assert_equal(model.predict_components([row])['points'],[20])

    def test_invalid_labels_baselines_dates_and_duplicates_rejected_before_fit(self):
        for rows in [[self.row(actual=-1)],[self.row(baseline=-1)],[self.row(actual=float('nan'))],
                     [self.row('2026-02-01')],[self.row(),self.row()]]:
            budget=FitBudget(1)
            with self.assertRaises(ValueError): ResidualOffsetModel({},1).fit(rows,budget)
            self.assertEqual(budget.count,0)

    def test_nonfinite_delta_or_unknown_multiplier_rejected(self):
        model=ResidualOffsetModel({},1); model.model=RecordingModel(float('nan'))
        with self.assertRaises(ValueError): model.predict_components([self.row()])
        with self.assertRaises(ValueError): ResidualOffsetModel({},3)
