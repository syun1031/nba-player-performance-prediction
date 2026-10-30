import unittest
from scripts.fetch_historical_seasons import compact_game,minutes_seconds,regular_cards


class HistoricalFetchTests(unittest.TestCase):
    def test_minutes_rounding_and_dnp(self):
        self.assertEqual(minutes_seconds('00:29'),29)
        self.assertEqual(minutes_seconds('12:30'),750)
        self.assertEqual(minutes_seconds('13:60'),840)
        self.assertIsNone(minutes_seconds('DNP - Coach decision'))
        for value in ['12:70','-1:00','bad']:
            with self.subTest(value=value),self.assertRaises(ValueError): minutes_seconds(value)

    def test_daily_cards_exclude_cup_final_and_other_seasons(self):
        p={'selectedDate':'2023-12-09','gameCardFeed':{'modules':[{'cards':[{'cardData':{'gameId':gid}} for gid in ['0022301230','0062300001','0042300001','0022400001']]}]}}
        self.assertEqual(regular_cards(p,'2023-12-09','2023-24'),['0022301230'])
        with self.assertRaises(ValueError):regular_cards(p,'2023-12-08','2023-24')

    def test_game_identity_and_roster_validation(self):
        def team(code):
            return dict(teamTricode=code,teamId=1,score=10,players=[dict(personId=i+1+(10 if code=='BBB' else 0),firstName='P',familyName=str(i),position='G' if i<5 else '',statistics={'minutes':'10:00','points':2,'reboundsTotal':1,'assists':0}) for i in range(5)])
        g=dict(gameId='0022400001',gameEt='2024-10-22T10:00:00Z',gameStatus=3,isNeutral=False,homeTeam=team('AAA'),awayTeam=team('BBB'),arena={'arenaName':'Arena'})
        c=compact_game(g,'0022400001','2024-25')
        self.assertEqual(len(c['homeTeam']['players']),5)
        with self.assertRaises(ValueError):compact_game(g,'0022400002','2024-25')
        g['homeTeam']['players'][0]['position']=''
        with self.assertRaises(ValueError):compact_game(g,'0022400001','2024-25')


if __name__=='__main__':unittest.main()
