import unittest
import numpy as np

from scripts.final_test_evaluation import validate_test_rows, frozen_model_check, evaluation_rows, high_score_groups


def sample(day='2026-03-01',actual='30'):
    return dict(player_id='P',game_id='AAA VS. BBB|'+day,GAMEDATE=day,split='test',target_pts=actual,
                pts_mean_5='20',is_starter='1',teammate_unknown_count='0')


class FinalTestTests(unittest.TestCase):
    def test_test_dates_are_closed_at_both_ends(self):
        rows=[sample(),sample('2026-04-12')]
        validate_test_rows(rows)
        for day in ['2026-02-28','2026-04-13','2026-03-1']:
            with self.assertRaises(ValueError):
                validate_test_rows([sample(day)])

    def test_rejects_duplicate_key_and_non_test_labels(self):
        with self.assertRaises(ValueError):
            validate_test_rows([sample(),sample()])
        with self.assertRaises(ValueError):
            validate_test_rows([dict(sample(),split='validation')])

    def test_rejects_nonfinite_targets_and_negative_points(self):
        for value in ['nan','inf','-1']:
            with self.assertRaises(ValueError):
                validate_test_rows([sample(actual=value)])

    def test_frozen_model_rejects_later_training_and_feature_order_changes(self):
        bundle=dict(fit_end_date='2026-01-31',feature_columns=['pts_mean_5','is_starter'])
        frozen_model_check(bundle,['pts_mean_5','is_starter'])
        with self.assertRaises(ValueError):
            frozen_model_check(dict(bundle,fit_end_date='2026-02-28'),bundle['feature_columns'])
        with self.assertRaises(ValueError):
            frozen_model_check(bundle,['is_starter','pts_mean_5'])

    def test_high_score_groups_are_inclusive_and_nested(self):
        rows=evaluation_rows([sample('2026-03-01','19'),sample('2026-03-02','20'),
                              sample('2026-03-03','29'),sample('2026-03-04','30')],[10,10,10,10])
        groups=high_score_groups(rows)
        self.assertEqual([(g['group'],g['n']) for g in groups],[('all',4),('actual_20_plus',3),('actual_30_plus',1)])
        self.assertEqual(groups[2]['mae'],20.)
        self.assertEqual(groups[2]['mean_residual'],20.)

    def test_prediction_lengths_and_nonfinite_are_rejected_not_silently_zipped(self):
        for values in [[],[1,2],[float('nan')]]:
            with self.assertRaises(ValueError):
                evaluation_rows([sample()],values)

    def test_signed_residual_preserves_large_miss(self):
        rows=evaluation_rows([sample(actual='40')],[12.5])
        self.assertEqual(rows[0]['residual'],27.5)
        self.assertEqual(rows[0]['baseline_residual'],20.)


if __name__=='__main__':
    unittest.main()
