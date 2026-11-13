import copy
from datetime import datetime,timezone
import unittest
from tests import test_fetch_current_season as fetch_fixtures
from scripts.fetch_current_season import final_overtimes, resolve_final_status, validate_final_box


class FinalResolutionTests(unittest.TestCase):
    def fixture(self):
        g=fetch_fixtures.CurrentSeasonTests().game()
        g.update(gameStatus=1,gameStatusText='Final',period=4,gameClock='PT00M00.00S')
        for side in ['homeTeam','awayTeam']:
            for p in g[side]['players'][:5]:p['statistics']['minutes']='48:00'
        l=dict(gameId=g['gameId'],gameStatus=3,gameStatusText='Final')
        for side in ['homeTeam','awayTeam']:
            t=g[side]
            l[side]=dict(teamId=t['teamId'],teamAbbreviation=t['teamTricode'],players=[
                dict(personId=p['personId'],teamId=t['teamId'],lineupStatus='Confirmed',rosterStatus='Active',position=p['position']) for p in t['players']])
        return g,l,datetime(2026,11,3,tzinfo=timezone.utc)

    def test_conflicting_numeric_status_requires_independent_final_and_full_box(self):
        g,l,now=self.fixture();before=copy.deepcopy(g)
        for code in [1,2]:
            g['gameStatus']=code;r=resolve_final_status(g,l,now)
            self.assertEqual(r['gameStatus'],3)
            self.assertEqual(g['gameStatus'],code)
        self.assertEqual(g['homeTeam'],before['homeTeam'])

    def test_reject_unconfirmed_nonfinal_or_wrong_reference(self):
        for field in ['gameStatus','gameId','team','starter','confirmed']:
            g,l,now=self.fixture()
            if field=='gameStatus':l['gameStatus']=2
            if field=='gameId':l['gameId']='0022609999'
            if field=='team':l['homeTeam']['teamId']=999
            if field=='starter':l['homeTeam']['players'][0]['personId']=999
            if field=='confirmed':l['homeTeam']['players'][0]['lineupStatus']='Projected'
            with self.subTest(field=field),self.assertRaises(ValueError):resolve_final_status(g,l,now)

    def test_reject_missing_clock_bad_minutes_scores_and_future_start(self):
        for bad in ['clock','period','minute','score','start','text','missing_minutes']:
            g,l,now=self.fixture()
            if bad=='clock':g['gameClock']='PT01M00.00S'
            if bad=='period':g['period']=3
            if bad=='minute':g['homeTeam']['players'][0]['statistics']['minutes']='47:00'
            if bad=='missing_minutes':g['homeTeam']['players'][0]['statistics']['minutes']=None
            if bad=='score':g['homeTeam']['score']=99
            if bad=='start':g['gameTimeUTC']='2026-11-04T23:00:00Z'
            if bad=='text':g['gameStatusText']='Q4 00:00'
            with self.subTest(bad=bad),self.assertRaises(ValueError):resolve_final_status(g,l,now)

    def test_per_player_whole_second_precision_allows_small_total_drift(self):
        g,l,now=self.fixture()
        g['homeTeam']['players'][0]['statistics']['minutes']='47:57'
        self.assertEqual(resolve_final_status(g,l,now)['gameStatus'],3)

    def test_final_label_overtime_forms(self):
        cases={'Final':0,'Final/OT':1,'Final/OT2':2,'Final/2OT':2,'Final/OT3':3,' Final/OT2 ':2}
        for text,expected in cases.items():
            with self.subTest(text=text):self.assertEqual(final_overtimes(text),expected)
        for text in ['Q4 00:00','Final/OT0','Final/0OT','Final/','Final OT2','final','',None]:
            with self.subTest(text=text):self.assertIsNone(final_overtimes(text))

    def double_overtime_box(self):
        # 0022600228 (2026-11-12) was published as "Final/OT2" after six periods.
        g,_,now=self.fixture()
        g.update(gameStatus=3,gameStatusText='Final/OT2',period=6)
        for side in ['homeTeam','awayTeam']:
            for p in g[side]['players'][:5]:p['statistics']['minutes']='58:00'
        return g,now

    def test_double_overtime_final_label_is_accepted(self):
        g,now=self.double_overtime_box()
        validate_final_box(g,now)
        g['gameStatusText']='Final/2OT';validate_final_box(g,now)

    def test_final_label_must_agree_with_period(self):
        for text,period in [('Final/OT2',5),('Final',6),('Final/OT',6),('Final/OT3',6)]:
            g,now=self.double_overtime_box();g.update(gameStatusText=text,period=period)
            with self.subTest(text=text,period=period),self.assertRaisesRegex(ValueError,'overtime count disagrees|team minutes'):validate_final_box(g,now)


if __name__=='__main__':unittest.main()
