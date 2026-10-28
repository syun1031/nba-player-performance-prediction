import json
import unittest
from scripts.teammate_encoder import TeammateEncoder


class EncodingTests(unittest.TestCase):
    def test_multi_hot_order_invariant(self):
        encoder=TeammateEncoder().fit([['B','A'],['C']])
        self.assertEqual(encoder.classes_,['A','B','C'])
        self.assertEqual(encoder.transform([['A','C']]),[[1,0,1,0]])
        self.assertEqual(encoder.transform([['C','A']]),[[1,0,1,0]])

    def test_unknowns_counted_without_expanding_vocabulary(self):
        encoder=TeammateEncoder().fit([['A','B']])
        before=encoder.to_dict()
        self.assertEqual(encoder.transform([['A','New','Other']]),[[1,0,2]])
        self.assertEqual(encoder.to_dict(),before)

    def test_transform_before_fit_rejected(self):
        with self.assertRaises(ValueError):
            TeammateEncoder().transform([['A']])

    def test_names_are_not_single_string_or_missing_or_duplicates(self):
        for names in ['ABC',None,['A','A'],['A',''],[' A']]:
            with self.assertRaises(ValueError):
                TeammateEncoder().fit([names])

    def test_restore_same_columns_and_values(self):
        encoder=TeammateEncoder().fit([['Luka Dončić','A']])
        state=json.loads(json.dumps(encoder.to_dict(),ensure_ascii=False))
        restored=TeammateEncoder.from_dict(state)
        self.assertEqual(encoder.get_feature_names_out(),restored.get_feature_names_out())
        self.assertEqual(encoder.transform([['Luka Dončić','Unknown']]),
                         restored.transform([['Luka Dončić','Unknown']]))

    def test_refit_resets_previous_classes(self):
        encoder=TeammateEncoder().fit([['Old']])
        encoder.fit([['New']])
        self.assertEqual(encoder.classes_,['New'])
        self.assertEqual(encoder.transform([['Old']]),[[0,1]])

    def test_four_or_five_teammates_preserve_total(self):
        encoder=TeammateEncoder().fit([['A','B','C']])
        values=encoder.transform([['A','B','C','D'],['A','B','C','D','E']])
        self.assertEqual([sum(v) for v in values],[4,5])
        self.assertEqual(encoder.get_feature_names_out()[-1],'teammate_unknown_count')

    def test_corrupt_persisted_classes_rejected(self):
        encoder=TeammateEncoder().fit([['A','B']])
        state=encoder.to_dict()
        state['classes']=['A','A']
        with self.assertRaises(ValueError):
            TeammateEncoder.from_dict(state)


if __name__=='__main__':
    unittest.main()
