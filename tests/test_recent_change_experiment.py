import unittest
from scripts.experiment_recent_changes import development_history,augment_development_rows,group_scores


def raw(day,pts=10,minutes=20):
    return dict(PLAYER='Player',TEAM='AAA',MATCHUP='AAA @ BBB',GAMEDATE=day,PTS=str(pts),MIN=str(minutes),REB='4',AST='2')


class RecentChangeExperimentTests(unittest.TestCase):
    def fixture(self):
        history=development_history([raw(f'2025-10-{d:02d}',p,m) for d,p,m in
                                    zip(range(21,26),[0,10,20,30,40],[10,20,30,40,50])]+[raw('2025-10-27',99)])
        sample=dict(player_id='Player',game_id='AAA @ BBB|2025-10-27',GAMEDATE='2025-10-27',split='train',
                    target_pts='99',pts_mean_5='20',pts_mean_10='20',min_mean_5='30',reb_mean_5='4',ast_mean_5='2',
                    is_home='0',is_neutral_site='0',rest_days='1',is_starter='1')
        return history,sample

    def test_raw_test_dates_are_skipped_before_outcomes_are_accessed(self):
        class ForbiddenOutcome(dict):
            def __getitem__(self,key):
                if key in ['PTS','MIN','REB','AST']:
                    raise AssertionError('Test outcomes must not be used')
                return super().__getitem__(key)
        result=development_history([raw('2025-10-21'),ForbiddenOutcome(raw('2026-03-01'))])
        self.assertEqual(len(result),1)

    def test_augmentation_preserves_nine_features_keys_and_labels(self):
        history,sample=self.fixture()
        result=augment_development_rows([sample],history)
        self.assertEqual(len(result),1)
        self.assertTrue(all(result[0][k]==v for k,v in sample.items()))
        self.assertEqual(result[0]['pts_last_1'],40.)
        self.assertEqual(result[0]['min_mean_3'],40.)

    def test_augmentation_rejects_test_targets_and_mismatched_existing_features(self):
        history,sample=self.fixture()
        for row in [dict(sample,GAMEDATE='2026-03-01',split='test'),dict(sample,pts_mean_5='999')]:
            with self.assertRaises(ValueError):
                augment_development_rows([row],history)

    def test_duplicate_samples_and_missing_target_source_are_rejected(self):
        history,sample=self.fixture()
        with self.assertRaises(ValueError):
            augment_development_rows([sample,sample],history)
        with self.assertRaises(ValueError):
            augment_development_rows([sample],history[:-1])

    def test_group_metrics_use_same_samples_and_keep_empty_group_unscored(self):
        result=group_scores([0,20,29],[5,19,25],[4,15,20])
        self.assertEqual([r['n'] for r in result],[3,2,0])
        self.assertEqual(result[1]['mae'],2.5)
        self.assertEqual(result[1]['baseline_mae'],7.)
        self.assertEqual(result[1]['mean_residual'],2.5)
        self.assertIsNone(result[2]['mae'])


if __name__=='__main__':
    unittest.main()
