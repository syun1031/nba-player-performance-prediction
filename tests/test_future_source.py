import copy
import unittest
from datetime import datetime,timezone,timedelta
from nba_dashboard.future_source import assess_context,parse_schedule


NOW=datetime(2026,11,4,23,40,tzinfo=timezone.utc)


def fixture():
    s=dict(nba_game_id='0022600174',GAMEDATE='2026-11-04',home_team='WAS',away_team='ATL',
           home_team_id=1,away_team_id=2,scheduled_start_utc='2026-11-05T00:00:00Z',game_status=1)
    def team(code,tid):
        return dict(teamId=tid,teamAbbreviation=code,players=[
            dict(personId=tid*100+i,teamId=tid,playerName=f'{code} {i}',lineupStatus='Confirmed',
                 rosterStatus='Active' if i<6 else 'Inactive',position=['PG','SG','SF','PF','C'][i] if i<5 else '',
                 timestamp='2026-11-04T23:25:00') for i in range(7)])
    game=dict(gameId=s['nba_game_id'],gameStatus=1,homeTeam=team('WAS',1),awayTeam=team('ATL',2))
    box=dict(gameId=s['nba_game_id'],gameStatus=1,gameEt='2026-11-04T19:00:00Z',
             gameTimeUTC=s['scheduled_start_utc'],isNeutral=False,
             homeTeam=dict(teamTricode='WAS',teamId=1),awayTeam=dict(teamTricode='ATL',teamId=2))
    def capture(data,url):return dict(data=data,url=url,observed_at_utc=NOW.isoformat(),sha256='a'*64)
    return s,capture(box,'https://www.nba.com/game/0022600174/box-score'),capture({'games':[game]},'https://stats.nba.com/js/data/leaders/00_daily_lineups_20261104.json')


class FutureSourceTests(unittest.TestCase):
    def test_confirmed_five_and_active_bench(self):
        c=assess_context(*fixture(),NOW)
        self.assertTrue(c['ready'],c['reasons'])
        self.assertEqual(len(c['teams']['WAS']['starters']),5)
        self.assertEqual(len(c['teams']['WAS']['active']),6)
        self.assertEqual(c['teams']['WAS']['players'][0]['source_timestamp_raw'],'2026-11-04T23:25:00')

    def test_projected_and_missing_game_never_unlock(self):
        s,b,l=fixture();l['data']['games'][0]['homeTeam']['players'][0]['lineupStatus']='Projected'
        self.assertFalse(assess_context(s,b,l,NOW)['ready'])
        l['data']['games']=[]
        self.assertFalse(assess_context(s,b,l,NOW)['ready'])

    def test_game_already_started_even_if_confirmed(self):
        s,b,l=fixture()
        self.assertFalse(assess_context(s,b,l,NOW+timedelta(minutes=20))['ready'])
        b['data']['gameStatus']=2
        self.assertFalse(assess_context(s,b,l,NOW)['ready'])
        b['data']['gameStatus']=1;l['data']['games'][0]['gameStatus']=3
        self.assertFalse(assess_context(s,b,l,NOW)['ready'])

    def test_window_inclusive_at_30_minutes_and_no_earlier(self):
        s,b,l=fixture();t=NOW-timedelta(minutes=10)
        for src in [b,l]:src['observed_at_utc']=t.isoformat()
        self.assertTrue(assess_context(s,b,l,t)['ready'])
        for src in [b,l]:src['observed_at_utc']=(t-timedelta(seconds=1)).isoformat()
        self.assertFalse(assess_context(s,b,l,t-timedelta(seconds=1))['ready'])

    def test_expired_or_future_observation(self):
        s,b,l=fixture();b['observed_at_utc']=(NOW-timedelta(minutes=3)).isoformat()
        self.assertFalse(assess_context(s,b,l,NOW)['ready'])
        b['observed_at_utc']=(NOW+timedelta(seconds=1)).isoformat()
        self.assertFalse(assess_context(s,b,l,NOW)['ready'])

    def test_wrong_team_player_id_and_start_time(self):
        for mutation in ['team','player','start','date','source']:
            s,b,l=fixture()
            if mutation=='team':l['data']['games'][0]['homeTeam']['teamId']=99
            if mutation=='player':l['data']['games'][0]['homeTeam']['players'][0]['personId']=200
            if mutation=='start':b['data']['gameTimeUTC']='2026-11-05T01:00:00Z'
            if mutation=='date':b['data']['gameEt']='2026-11-05T19:00:00Z'
            if mutation=='source':l['url']='https://example.com/lineups.json'
            with self.subTest(mutation=mutation):self.assertFalse(assess_context(s,b,l,NOW)['ready'])

    def test_missing_starter_or_inactive_starter(self):
        for change in ['position','rosterStatus']:
            s,b,l=fixture();p=l['data']['games'][0]['awayTeam']['players'][0]
            p[change]='' if change=='position' else 'Inactive'
            self.assertFalse(assess_context(s,b,l,NOW)['ready'])

    def test_schedule_excludes_other_seasons_and_cup_final(self):
        def card(gid):return dict(gameId=gid,seasonYear='2026-27',seasonType='Regular Season',gameStatus=1,
            gameTimeEastern='2026-11-04T19:00:00Z',gameTimeUtc='2026-11-05T00:00:00Z',
            homeTeam=dict(teamTricode='WAS',teamId=1),awayTeam=dict(teamTricode='ATL',teamId=2))
        p=dict(selectedDate='2026-11-04',gameCardFeed=dict(modules=[dict(cards=[dict(cardData=card(g)) for g in ['0022600174','0062600001','0022500001']])]))
        self.assertEqual(len(parse_schedule(p,'2026-11-04')),1)
        p['selectedDate']='2026-11-05'
        with self.assertRaises(ValueError):parse_schedule(p,'2026-11-04')


if __name__=='__main__':unittest.main()
