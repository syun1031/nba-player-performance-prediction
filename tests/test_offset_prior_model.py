import unittest
from unittest.mock import patch
import numpy as np
from scripts.offset_prior_model import OffsetPriorModel,COLUMNS
from scripts.time_validation_tuning import FitBudget


class FakeRegressor:
    def fit(self,x,y,sample_weight):self.x=x;self.y=y;self.weights=sample_weight
    def predict(self,x):return np.array([-30.,40.])[:len(x)]


def row(name='A',points=35,base=30,day='2025-11-01'):
    r={c:1 for c in COLUMNS}
    r.update(player_id=name,game_id=name+'|'+day,GAMEDATE=day,split='train',target_pts=points,pts_mean_5=base,
        has_prev_season=0,prev_season_games=0,prev_season_pts_mean=None,prev_season_min_mean=None,prev_season_pts_per_min=None,
        same_conference=0,same_conf_h2h_count=0,same_conf_h2h_pts_last_1=None,same_conf_h2h_pts_mean=None)
    return r


class OffsetPriorModelTests(unittest.TestCase):
    @patch('scripts.offset_prior_model.XGBRegressor',return_value=FakeRegressor())
    def test_signed_target_original_points_weights_and_nan_features(self,mock):
        budget=FitBudget(1);m=OffsetPriorModel({}).fit([row(),row('B',20,25)],budget)
        np.testing.assert_equal(m.model.y,[5,-5]);np.testing.assert_allclose(m.model.weights,[4/3,2/3])
        self.assertEqual(budget.count,1);self.assertEqual(m.model.x.shape,(2,25))
        self.assertTrue(np.isnan(m.model.x[:,COLUMNS.index('prev_season_pts_mean')]).all())
        self.assertEqual(m.train_high_n,1);self.assertEqual(m.weight_normalizer,1.5)

    @patch('scripts.offset_prior_model.XGBRegressor',return_value=FakeRegressor())
    def test_add_baseline_clip_only_final_and_ignore_current_labels(self,mock):
        m=OffsetPriorModel({}).fit([row(),row('B',20,25)],FitBudget(1))
        rows=[row(base=20),row('B',base=25)]
        a=m.predict_components(rows);np.testing.assert_equal(a['delta'],[-30,40]);np.testing.assert_equal(a['raw_points'],[-10,65]);np.testing.assert_equal(a['points'],[0,65])
        fake=[dict(r,target_pts=99999,PTS=99999,MIN=99999) for r in rows]
        for k,v in a.items():np.testing.assert_equal(v,m.predict_components(fake)[k])
        absent=[{k:v for k,v in r.items() if not k.startswith('target_')} for r in rows]
        np.testing.assert_equal(a['points'],m.predict_components(absent)['points'])

    def test_bad_training_does_not_consume_fit(self):
        good=row()
        for rows in [[row(day='2026-02-01')],[row(day='2024-11-01')],[dict(good,split='validation')],[good,good],
            [row(points=-1)],[row(base=-1)],[row(points=float('nan'))],[dict(good,min_mean_5=float('inf'))]]:
            budget=FitBudget(1)
            with self.subTest(rows=rows),self.assertRaises(ValueError):OffsetPriorModel({}).fit(rows,budget)
            self.assertEqual(budget.count,0)


if __name__=='__main__':unittest.main()
