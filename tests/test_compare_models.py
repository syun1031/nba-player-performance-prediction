import unittest
import numpy as np

from scripts.compare_models import validate_development_rows, feature_groups, paired_mae_interval


class ComparisonTests(unittest.TestCase):
    def rows(self):
        return [dict(player_id='P',game_id='G',GAMEDATE='2026-01-31',split='train')], [
            dict(player_id='P',game_id='H',GAMEDATE='2026-02-01',split='validation')]

    def test_rejects_test_rows_and_wrong_date_even_if_split_label_looks_right(self):
        train,val=self.rows()
        for split,date in [('test','2026-03-01'),('validation','2026-03-01'),('validation','2026-01-31')]:
            bad=[dict(val[0],split=split,GAMEDATE=date)]
            with self.assertRaises(ValueError):
                validate_development_rows(train,bad)

    def test_rejects_duplicate_keys_and_overlapping_samples(self):
        train,val=self.rows()
        with self.assertRaises(ValueError):
            validate_development_rows(train+train,val)
        with self.assertRaises(ValueError):
            validate_development_rows(train,[dict(val[0],game_id='G')])

    def test_features_never_include_target_or_ids(self):
        groups=feature_groups(['pts_mean_5','is_starter','teammate_0001','teammate_unknown_count'])
        self.assertEqual(groups,dict(basic=['pts_mean_5'],starter=['pts_mean_5','is_starter'],
                                    teammates=['pts_mean_5','is_starter','teammate_0001','teammate_unknown_count']))
        for bad in ['target_pts','player_id','game_id','GAMEDATE','split','pregame_verified','unknown_column']:
            with self.assertRaises(ValueError):
                feature_groups(['pts_mean_5','is_starter',bad])

    def test_paired_mae_improvement_is_negative_and_reproducible(self):
        y=np.array([0.,0.,0.,0.]); old=np.array([2.,2.,2.,2.]); new=np.array([1.,1.,1.,1.])
        dates=['2026-02-01','2026-02-01','2026-02-02','2026-02-02']
        result=paired_mae_interval(y,old,new,dates,iterations=100)
        self.assertEqual(result,paired_mae_interval(y,old,new,dates,iterations=100))
        self.assertEqual(result['delta_mae'],-1.)
        self.assertEqual(result['lower_95'],-1.)
        self.assertEqual(result['upper_95'],-1.)

    def test_cluster_interval_preserves_all_samples_within_sampled_date(self):
        # Day A: 3 rows improve by 1; day B: one row worsens by 3.
        result=paired_mae_interval([0]*4,[1,1,1,0],[0,0,0,3],['A','A','A','B'],iterations=1000)
        self.assertEqual(result['delta_mae'],0.)
        self.assertEqual(result['lower_95'],-1.)
        self.assertEqual(result['upper_95'],3.)

    def test_rejects_nonfinite_misaligned_and_single_date_interval_inputs(self):
        for y,old,new,dates in [([1],[1],[1,2],['A']),([1,2],[1,2],[1,float('nan')],['A','B']),
                                ([1,2],[1,2],[1,2],['A','A'])]:
            with self.assertRaises(ValueError):
                paired_mae_interval(y,old,new,dates)


if __name__=='__main__':
    unittest.main()
