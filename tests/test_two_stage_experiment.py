import unittest
from scripts.experiment_two_stage import control_predictions,minute_groups


class TwoStageExperimentTests(unittest.TestCase):
    def test_control_join_follows_requested_order_and_checks_labels(self):
        rows=[dict(player_id=p,game_id=p+'|2026-02-01',target_pts=y) for p,y in [('B',0),('A',10)]]
        controls=[dict(rows[1],pred_v1=9,pred_v2=8),dict(rows[0],pred_v1=2,pred_v2=1)]
        self.assertEqual(control_predictions(rows,controls,'pred_v1').tolist(),[2,9])
        for bad in [controls[:1],controls+[controls[0]],
                    [dict(controls[0],target_pts=999),controls[1]],
                    [dict(controls[0],pred_v1='nan'),controls[1]]]:
            with self.assertRaises(ValueError):
                control_predictions(rows,bad,'pred_v1')

    def test_minute_groups_boundaries_and_change_threshold(self):
        rows=[dict(target_minutes=m,min_mean_5=20) for m in [1,10,20,30,53]]
        masks=minute_groups(rows)
        self.assertEqual([masks[g].sum() for g in ['under10','10_to_under20','20_to_under30','30_plus']],[1,1,1,2])
        self.assertEqual(masks['minutes_change_10_plus'].tolist(),[True,True,False,True,True])


if __name__=='__main__':
    unittest.main()
