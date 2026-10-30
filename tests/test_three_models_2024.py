import unittest
from scripts.three_models_2024 import base_matrix,fit_base,BASE
from scripts.time_validation_tuning import FitBudget


class ThreeModelsTests(unittest.TestCase):
    def test_nine_only_ignores_profile_and_target(self):
        row={c:1 for c in BASE}
        x=base_matrix([row]);self.assertEqual(x.shape,(1,9))
        y=base_matrix([dict(row,target_pts='SEALED',PTS='SEALED',prev_season_pts_mean='IGNORED')])
        self.assertTrue((x==y).all())

    def test_invalid_training_does_not_consume_budget(self):
        for extra in [dict(GAMEDATE='2025-02-01',split='train'),dict(GAMEDATE='2024-11-01',split='validation'),dict(GAMEDATE='2024-11-01',split='train',target_pts=-1)]:
            row=dict({c:1 for c in BASE},player_id='A',game_id='x',target_pts=10,**{k:v for k,v in extra.items() if k!='target_pts'})
            if 'target_pts' in extra:row['target_pts']=extra['target_pts']
            budget=FitBudget(1)
            with self.assertRaises(ValueError):fit_base([row],{},budget)
            self.assertEqual(budget.count,0)


if __name__=='__main__':unittest.main()
