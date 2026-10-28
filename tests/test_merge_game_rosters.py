import unittest
from scripts.merge_game_rosters import merge_records


def roster():
    return dict(MATCHUP='AAA @ BBB', GAMEDATE='2026-01-01', TEAM='BBB',
                **{f'PLAYER_{i+1}': name for i, name in enumerate(
                    ['*Bobby Portis', '*B', '*C', '*D', '*E', 'Reserve', 'Zero'])})


def log(name='Bobby Portis Jr.', minutes='20', matchup='BBB VS. AAA'):
    return dict(PLAYER=name, TEAM='BBB', MATCHUP=matchup,
                GAMEDATE='2026-01-01', MIN=minutes, PTS='0',
                **{'FG%': '-', 'REB': '0', 'AST': '0'})


class MergeTests(unittest.TestCase):
    def test_alias_starter_and_preserve_complete_roster(self):
        rows = merge_records([log()], [roster()])
        self.assertEqual(rows[0]['is_starter'], 1)
        self.assertEqual(rows[0]['game_id'], 'BBB VS. AAA|2026-01-01')
        self.assertEqual(len(rows[0]['starting_teammates']), 4)
        self.assertIn('Zero', rows[0]['appeared_players'])
        self.assertEqual(rows[0]['PLAYER'], 'Bobby Portis Jr.')

    def test_reserve_has_five_starting_teammates(self):
        row = merge_records([log('Reserve')], [roster()])[0]
        self.assertEqual(row['is_starter'], 0)
        self.assertEqual(len(row['starting_teammates']), 5)

    def test_zero_is_excluded_not_removed_from_roster(self):
        row = merge_records([log('Zero', '0')], [roster()])[0]
        self.assertEqual(row['exclusion_reason'], 'zero_minutes')
        self.assertFalse(row['merge_eligible'])

    def test_direction_conflict_has_unknown_home(self):
        row = merge_records([log(matchup='BBB @ AAA')], [roster()])[0]
        self.assertEqual(row['home_away_conflict'], 1)
        self.assertIsNone(row['is_home'])
        self.assertFalse(row['merge_eligible'])

    def test_different_opponent_must_fail(self):
        with self.assertRaises(ValueError):
            merge_records([log(matchup='BBB @ CCC')], [roster()])

    def test_positive_minutes_missing_player_must_fail(self):
        with self.assertRaises(ValueError):
            merge_records([log('Unknown')], [roster()])

    def test_duplicate_key_must_fail(self):
        with self.assertRaises(ValueError):
            merge_records([log(), log()], [roster()])

    def test_verified_neutral_applies_to_both_teams_and_resolves_conflict(self):
        venue = dict(GAMEDATE='2026-01-01', teams=['AAA', 'BBB'],
                     venue='Neutral Arena', source_url='https://www.nba.com/example',
                     designated_home_team='BBB')
        other = dict(roster(), TEAM='AAA')
        away = dict(log('Reserve'), TEAM='AAA', MATCHUP='AAA @ BBB')
        # The first row has a direction conflict; both rows belong to the same venue.
        rows = merge_records([log(matchup='BBB @ AAA'), away], [roster(), other], venues=[venue])
        self.assertEqual([r['is_neutral_site'] for r in rows], [1, 1])
        self.assertEqual([r['is_home'] for r in rows], [0, 0])
        self.assertTrue(all(r['merge_eligible'] for r in rows))
        self.assertEqual(rows[0]['home_away_conflict'], 1)
        self.assertEqual(rows[0]['home_away_resolution'], 'verified_neutral_site')
        self.assertEqual(rows[0]['designated_home_team'], 'BBB')
        self.assertEqual(rows[0]['game_id'], 'BBB @ AAA|2026-01-01')

    def test_unverified_venue_not_assumed_non_neutral(self):
        row = merge_records([log()], [roster()], venues=[])[0]
        self.assertIsNone(row['is_neutral_site'])
        self.assertEqual(row['is_home'], 1)
        self.assertEqual(row['home_away_resolution'], 'source_agreement')

    def test_neutral_cannot_override_different_opponents(self):
        venue = dict(GAMEDATE='2026-01-01', teams=['AAA', 'CCC'],
                     venue='Other Arena', source_url='https://www.nba.com/example',
                     designated_home_team=None)
        row = merge_records([log(matchup='BBB @ AAA')], [roster()], venues=[venue])[0]
        self.assertIsNone(row['is_neutral_site'])
        self.assertIsNone(row['is_home'])
        self.assertFalse(row['merge_eligible'])

    def test_official_non_neutral_has_zero_flag_and_official_home(self):
        venue = dict(GAMEDATE='2026-01-01', teams=['AAA', 'BBB'],
                     venue='Home Arena', source_url='https://www.nba.com/example',
                     designated_home_team='BBB', is_neutral_site=False)
        row = merge_records([log()], [roster()], venues=[venue])[0]
        self.assertEqual(row['is_neutral_site'], 0)
        self.assertEqual(row['is_home'], 1)
        self.assertEqual(row['home_away_resolution'], 'verified_official_venue')

    def test_non_neutral_without_official_home_must_fail(self):
        venue = dict(GAMEDATE='2026-01-01', teams=['AAA', 'BBB'],
                     venue='Home Arena', source_url='https://www.nba.com/example',
                     designated_home_team=None, is_neutral_site=False)
        with self.assertRaises(ValueError):
            merge_records([log()], [roster()], venues=[venue])


if __name__ == '__main__':
    unittest.main()
