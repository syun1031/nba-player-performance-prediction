import unittest
from scripts.offset_prior_2024 import OffsetPrior2024Model
from scripts.time_validation_tuning import FitBudget
from tests.test_offset_prior_model import row,FakeRegressor
from unittest.mock import patch

class Offset2024Tests(unittest.TestCase):
    def test_2024_training_guard_and_signed_target(self):
        rows=[dict(row('a',35,30),GAMEDATE='2024-11-01'),dict(row('b',10,15),GAMEDATE='2025-01-31')]
        with patch('scripts.offset_prior_2024.XGBRegressor',return_value=FakeRegressor()):
            model=OffsetPrior2024Model({}).fit(rows,FitBudget(1))
        self.assertEqual(model.trained_through,'2025-01-31')
        self.assertEqual(model.model.y.tolist(),[5,-5])
        self.assertEqual(model.train_high_n,1)
        for day in ['2024-10-21','2025-02-01','2025-11-01']:
            budget=FitBudget(1)
            with self.assertRaises(ValueError):OffsetPrior2024Model({}).fit([dict(rows[0],GAMEDATE=day)],budget)
            self.assertEqual(budget.count,0)
