import unittest
from scripts.compare_offset_prior_2025 import align_controls


class OffsetPriorComparisonTests(unittest.TestCase):
    def row(self,p='A',**kwargs):
        return dict(player_id=p,game_id=p+' @ B|2026-03-01',GAMEDATE='2026-03-01',period='Test',target_pts=30,pred_baseline=20,**kwargs)

    def test_control_alignment_follows_keys_not_row_order(self):
        a=[self.row('A',pred_residual_w1=21),self.row('C',pred_residual_w1=22)]
        b=[self.row('C',pred_basic_prior14=25),self.row('A',pred_basic_prior14=24)]
        pairs=align_controls(a,b)
        self.assertEqual([r['player_id'] for r,s in pairs],['A','C'])
        self.assertEqual([s['pred_basic_prior14'] for r,s in pairs],[24,25])

    def test_reject_different_cohort_labels_dates_baseline_and_fold(self):
        row=self.row()
        for other in [self.row('C'),dict(row,target_pts=31),dict(row,GAMEDATE='2026-03-02'),dict(row,pred_baseline=21),dict(row,period='F1')]:
            with self.assertRaises(ValueError):align_controls([row],[other])
        with self.assertRaises(ValueError):align_controls([row],[row,row])
        with self.assertRaises(ValueError):align_controls([row,row],[row])


if __name__=='__main__':unittest.main()
