import unittest
import math

from scripts.analyze_residuals import build_residuals, summarize, score_band, grouped_bands, player_profiles, choose_cases


class ResidualTests(unittest.TestCase):
    def fixtures(self):
        predictions=[dict(player_id='P',game_id='AAA VS. BBB|2026-02-01',GAMEDATE='2026-02-01',
                          target_pts='10',pred_model='8',pred_baseline='12',teammate_unknown_count='0')]
        features=[dict(predictions[0],split='validation',pts_mean_5='12',is_starter='1')]
        return predictions,features

    def test_residual_sign_matches_under_and_overprediction(self):
        pred,features=self.fixtures()
        rows=build_residuals(pred,features,'pred_model')
        self.assertEqual(rows[0]['residual'],2.)
        self.assertEqual(rows[0]['baseline_residual'],-2.)
        self.assertEqual(rows[0]['absolute_error'],2.)

    def test_join_rejects_missing_duplicate_keys_wrong_labels_and_test_dates(self):
        pred,features=self.fixtures()
        for p,f in [(pred,[]),(pred+pred,features),(pred,features+features),
                    (pred,[dict(features[0],target_pts='11')]),
                    ([dict(pred[0],GAMEDATE='2026-03-01')],features)]:
            with self.assertRaises(ValueError):
                build_residuals(p,f,'pred_model')

    def test_nonfinite_and_incorrect_baseline_are_rejected(self):
        pred,features=self.fixtures()
        for row in [dict(pred[0],pred_model='nan'),dict(pred[0],pred_baseline='11')]:
            with self.assertRaises(ValueError):
                build_residuals([row],features,'pred_model')

    def test_metrics_include_large_misses_and_do_not_cancel_signed_errors(self):
        rows=[dict(actual=a,predicted=p,baseline=b,residual=a-p,absolute_error=abs(a-p),baseline_residual=a-b)
              for a,p,b in [(10.,8.,12.),(20.,15.,10.),(0.,5.,3.)]]
        result=summarize(rows)
        self.assertEqual(result['n'],3)
        self.assertEqual(result['mae'],4.)
        self.assertAlmostEqual(result['rmse'],math.sqrt(18))
        self.assertAlmostEqual(result['mean_residual'],2/3)
        self.assertEqual(result['within_5_rate'],1.)

    def test_bins_are_half_open_and_include_negative_predictions_without_clipping(self):
        self.assertEqual([score_band(v) for v in [-1,0,9.999,10,19.999,20,29.999,30]],
                         ['lt0','0_to_lt10','0_to_lt10','10_to_lt20','10_to_lt20','20_to_lt30','20_to_lt30','30_plus'])

    def test_empty_bands_have_no_fake_zero_errors(self):
        pred,features=self.fixtures()
        groups=grouped_bands(build_residuals(pred,features,'pred_model'),'actual')
        self.assertEqual(sum(r['n'] for r in groups),1)
        empty=next(r for r in groups if r['band']=='30_plus')
        self.assertEqual(empty['n'],0)
        self.assertIsNone(empty['mae'])
        self.assertIsNone(empty['mean_residual'])

    def test_case_rules_use_eight_game_floor_only_for_player_examples(self):
        profiles=[dict(player_id='Few',n=7,mae=.1,delta_mae=-9),
                  dict(player_id='Improved',n=8,mae=2.,delta_mae=-2.),
                  dict(player_id='Typical',n=10,mae=3.,delta_mae=0.),
                  dict(player_id='Worse',n=8,mae=5.,delta_mae=2.)]
        cases=choose_cases(profiles)
        self.assertEqual({r['role']:r['player_id'] for r in cases},
                         dict(most_improved='Improved',most_worsened='Worse',typical='Typical'))

    def test_profiles_keep_players_below_case_floor(self):
        pred,features=self.fixtures()
        profiles=player_profiles(build_residuals(pred,features,'pred_model'))
        self.assertEqual(len(profiles),1)
        self.assertEqual(profiles[0]['n'],1)
        self.assertFalse(profiles[0]['case_eligible'])


if __name__=='__main__':
    unittest.main()
