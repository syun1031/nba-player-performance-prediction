import unittest
import math
import numpy as np

from scripts.time_validation_tuning import (
    build_folds, prepare_features, rebuild_teammates, parameter_grid, pooled_metrics,
    choose_common_parameters, stratified_interval, FitBudget, fit_candidate, effect_diagnostics,
)


def sample(day, names=None, player='P'):
    return dict(player_id=player,game_id='AAA VS. BBB|'+day,GAMEDATE=day,
                is_starter='1',target_pts='1',_teammates=names or ['A','B','C','D'],
                pts_mean_5='1',pts_mean_10='1',min_mean_5='1',reb_mean_5='1',
                ast_mean_5='1',is_home='1',is_neutral_site='0',rest_days='1')


class TimeValidationTests(unittest.TestCase):
    def test_folds_keep_same_date_together_and_exclude_later_dates(self):
        rows=[sample(day,player=p) for day in ['2025-11-30','2025-12-01','2025-12-31',
                 '2026-01-01','2026-01-15','2026-01-16','2026-01-31'] for p in ['P','Q']]
        folds=build_folds(rows)
        self.assertEqual([(len(a),len(b)) for _,a,b in folds],[(2,4),(6,4),(10,4)])
        for _,train,val in folds:
            self.assertLess(max(r['GAMEDATE'] for r in train),min(r['GAMEDATE'] for r in val))
            self.assertTrue(all(sum(x['GAMEDATE']==r['GAMEDATE'] for x in val)==2 for r in val))
        with self.assertRaises(ValueError):
            build_folds(rows+[sample('2026-02-01')])

    def test_each_fold_vocabulary_uses_only_its_training_names(self):
        train=[sample('2025-11-30')]
        val=[sample('2025-12-01',['A','B','C','NEW'])]
        p=prepare_features(train,val)
        self.assertEqual(p['encoder'].classes_,['A','B','C','D'])
        self.assertEqual(p['unknown_validation'].tolist(),[1])
        self.assertEqual(p['X']['validation']['teammates'].shape,(1,14))
        self.assertEqual(p['X']['validation']['teammates'][0,-1],1)

    def test_nonfinite_features_are_rejected_without_dropping_samples(self):
        train=[sample('2025-11-30')]
        val=[dict(sample('2025-12-01'),pts_mean_5='nan')]
        with self.assertRaises(ValueError):
            prepare_features(train,val)

    def test_roster_join_excludes_self_and_rejects_conflicting_starter_flag(self):
        row=sample('2025-11-30')
        roster={('2025-11-30','AAA'):dict(players=['P','A','B','C','D'],
                  starters=['P','A','B','C','D'],pair=frozenset(['AAA','BBB']))}
        rebuilt=rebuild_teammates([row],roster)
        self.assertEqual(rebuilt[0]['_teammates'],['A','B','C','D'])
        with self.assertRaises(ValueError):
            rebuild_teammates([dict(row,is_starter='0')],roster)

    def test_pooled_metrics_use_all_sample_errors_instead_of_averaging_fold_rmse(self):
        result=pooled_metrics([([0],[6]),([0,0,0],[2,2,2])])
        self.assertEqual(result['n'],4)
        self.assertEqual(result['mae'],3.)
        self.assertAlmostEqual(result['rmse'],math.sqrt(12))

    def test_parameter_choice_ignores_other_feature_group_scores(self):
        rows=[dict(model='tree',parameter_id='tree_01',features='starter',mae=3.,rmse=4.),
              dict(model='tree',parameter_id='tree_02',features='starter',mae=4.,rmse=5.),
              dict(model='tree',parameter_id='tree_02',features='teammates',mae=1.,rmse=2.)]
        grids={'tree':[dict(parameter_id='tree_01',params=dict(max_depth=6,min_samples_leaf=30)),
                       dict(parameter_id='tree_02',params=dict(max_depth=4,min_samples_leaf=60))]}
        self.assertEqual(choose_common_parameters(rows,grids)['tree']['parameter_id'],'tree_01')
        rows[1].update(mae=3.,rmse=4.)
        self.assertEqual(choose_common_parameters(rows,grids)['tree']['parameter_id'],'tree_02')

    def test_grid_budget_stops_before_exceeding_cap(self):
        grids=parameter_grid()
        self.assertEqual([len(grids[n]) for n in ['linear','tree','xgboost']],[1,6,8])
        runs=[(f,n,p['parameter_id'],g) for f in range(3) for n,params in grids.items()
              for p in params for g in ['basic','starter','teammates']]
        self.assertEqual(len(set(runs))+9,144)
        budget=FitBudget(2)
        budget.consume(); budget.consume()
        with self.assertRaises(RuntimeError):
            budget.consume()
        self.assertEqual(budget.count,2)

    def test_stratified_bootstrap_preserves_each_fold(self):
        result=stratified_interval([0]*6,[1,1,1,1,0,0],[0,0,0,0,3,3],
                                   ['A','B','C','D','E','F'],['F1']*4+['F2']*2,iterations=100)
        self.assertAlmostEqual(result['delta_mae'],1/3)
        self.assertAlmostEqual(result['lower_95'],1/3)
        self.assertAlmostEqual(result['upper_95'],1/3)

    def test_linear_scaler_is_not_fitted_to_validation(self):
        train=[sample('2025-11-29'),dict(sample('2025-11-30'),pts_mean_5='3',target_pts='3')]
        val=[dict(sample('2025-12-01'),pts_mean_5='1001')]
        p=prepare_features(train,val)
        budget=FitBudget(1)
        model,pred=fit_candidate('linear',dict(fit_intercept=True),'basic',p,budget)
        self.assertEqual(model.steps[0][1].mean_[0],2.)
        self.assertEqual(int(model.steps[0][1].n_samples_seen_),2)
        self.assertEqual(len(pred),1)

    def test_effect_diagnostics_reports_single_period_gain_and_unknown_regression(self):
        paired=[dict(model='xgboost',old_features='starter',period=p,delta_mae=d,lower_95=lo,upper_95=hi)
                for p,d,lo,hi in [('F1',-.1,-.2,0),('F2',.1,0,.2),('F3',.1,0,.2),('February',-.1,-.2,-.01)]]
        sub=[dict(period='February',model_variant='xgboost_starter',group='has_unknown',mae=4.),
             dict(period='February',model_variant='xgboost_teammates',group='has_unknown',mae=4.2)]
        result=effect_diagnostics('xgboost',paired,sub)
        self.assertTrue(result['only_one_or_no_improving_fold'])
        self.assertTrue(result['unknown_february_worse'])
        self.assertAlmostEqual(result['unknown_february_delta_mae'],.2)
        self.assertEqual(result['february_direction'],'better')


if __name__=='__main__':
    unittest.main()
