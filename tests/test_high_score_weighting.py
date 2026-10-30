import unittest
import numpy as np

from scripts.high_score_weighting import high_score_weights,input_matrix,VARIANTS,alert_metrics,point_group_scores,paired_comparison
from scripts.experiment_recent_changes import BASE
from scripts.same_conference_h2h import H2H_FEATURES


class HighScoreWeightingTests(unittest.TestCase):
    def test_boundary_and_mean_one_weights_keep_relative_multiplier(self):
        weights,normalizer=high_score_weights([0,29,30,40],3)
        np.testing.assert_equal(weights,[.5,.5,1.5,1.5])
        self.assertEqual(normalizer,2)
        self.assertEqual(weights.mean(),1)

    def test_weight_one_and_single_class_do_not_change_relative_weights(self):
        for y,m in [([0,30],1),([30,40],3),([0,10],2)]:
            w,_=high_score_weights(y,m)
            np.testing.assert_equal(w,np.ones(len(y)))

    def test_invalid_labels_or_multipliers_are_rejected(self):
        for y,m in [([],2),([-1,30],2),([0,float('nan')],2),([[1,2]],2),([0,30],.5),([0,30],float('inf'))]:
            with self.assertRaises(ValueError): high_score_weights(y,m)

    def test_variants_isolate_nine_features_from_combined_twenty(self):
        self.assertEqual([len(v['columns']) for v in VARIANTS.values()],[9,9,20,20])
        row=dict.fromkeys(BASE,1.)|dict(target_pts=999,same_conf_h2h_pts_mean=float('nan'))
        np.testing.assert_equal(input_matrix([row],'basic_w2'),np.ones((1,9)))
        row['target_pts']=0
        np.testing.assert_equal(input_matrix([row],'basic_w3'),np.ones((1,9)))

    def test_combined_missing_h2h_remains_nan_not_zero(self):
        row=dict.fromkeys(VARIANTS['combined_w2']['columns'],1.)
        row.update(same_conf_h2h_count=0,same_conf_h2h_pts_last_1='',same_conf_h2h_pts_mean='')
        x=input_matrix([row],'combined_w2')
        self.assertTrue(np.isnan(x[0,-1]) and np.isnan(x[0,-2]))

    def test_alert_counts_use_fixed_thirty_threshold(self):
        r=alert_metrics([35,30,10,0],[32,25,31,5])
        self.assertEqual([r[k] for k in ['tp','fp','fn','tn']],[1,1,1,1])
        self.assertEqual(r['precision'],.5); self.assertEqual(r['recall'],.5)
        self.assertEqual(r['false_positive_rate'],.5); self.assertEqual(r['prevalence'],.5)

    def test_no_alert_precision_is_undefined_but_recall_is_zero(self):
        r=alert_metrics([0,30],[5,20])
        self.assertIsNone(r['precision']); self.assertEqual(r['recall'],0)
        self.assertEqual(r['f1'],0); self.assertEqual(r['predicted_high_n'],0)
        self.assertEqual(r['average_precision'],1)

    def test_no_actual_high_has_no_fabricated_recall_or_ap(self):
        r=alert_metrics([0,10],[30,20])
        self.assertIsNone(r['recall']); self.assertIsNone(r['average_precision'])
        self.assertEqual(r['precision'],0)

    def test_group_metrics_are_unweighted_and_keep_empty_groups_missing(self):
        scores=point_group_scores([0,30],[3,20])
        index={r['group']:r for r in scores}
        self.assertEqual(index['all']['mae'],6.5)
        self.assertEqual(index['under30']['mae'],3)
        self.assertEqual(index['30+']['mean_residual'],10)
        self.assertEqual(index['under30']['overprediction_fraction'],1)
        self.assertEqual(index['10_to_under20']['n'],0)
        self.assertIsNone(index['10_to_under20']['mae'])

    def test_rare_high_group_does_not_get_a_fake_interval(self):
        rows=[dict(period='F1',GAMEDATE='2026-01-01',target_pts=30,pred_a=20,pred_b=22),
              dict(period='F2',GAMEDATE='2026-01-02',target_pts=35,pred_a=20,pred_b=22)]
        result=paired_comparison(rows,'a','b','30+')
        self.assertEqual(result['status'],'insufficient_dates')
        self.assertIsNone(result['lower_95']); self.assertEqual(result['delta_mae'],-2)
