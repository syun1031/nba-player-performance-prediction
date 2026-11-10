import copy
import io
import tempfile
import unittest
from pathlib import Path
from scripts.actions_report import annotation, report_error
from scripts.fetch_current_season import completed_rows, schedule_rows, prior_profiles, conference_teams, box_page, describe_error


class CurrentSeasonTests(unittest.TestCase):
    def game(self):
        def team(code, offset):
            return dict(teamTricode=code, teamId=offset, score=10,
                        players=[dict(personId=offset+i, firstName='Player', familyName=str(offset+i),
                                      position='G' if i<5 else '', statistics=dict(minutes='10:00' if i<5 else '00:00',
                                      points=2 if i<5 else 0, reboundsTotal=1, assists=0)) for i in range(6)])
        return dict(gameId='0022600001', gameStatus=3, gameEt='2026-10-20T19:00:00Z',
                    gameTimeUTC='2026-10-20T23:00:00Z', isNeutral=False,
                    homeTeam=team('BOS',100), awayTeam=team('NYK',200))

    def test_positive_minutes_only_and_no_extra_stats(self):
        players,teams,excluded=completed_rows(self.game())
        self.assertEqual((len(players),len(teams),len(excluded)),(10,2,2))
        self.assertEqual(sum(r['is_starter'] for r in players),10)
        self.assertTrue(all(r['pregame_verified'] is False for r in players))
        self.assertEqual(players[0]['game_id'],'BOS VS. NYK|2026-10-20')
        self.assertEqual(teams[0]['points_allowed'],10)
        self.assertNotIn('FG%',players[0])

    def test_verified_name_variant_keeps_legacy_key_and_source_name(self):
        g=self.game();p=g['homeTeam']['players'][0]
        p.update(personId=1630619,firstName='Moussa',familyName='Cissé')
        players,_,_=completed_rows(g)
        self.assertEqual(players[0]['player_id'],'Moussa Cisse')
        self.assertEqual(players[0]['source_player_name'],'Moussa Cissé')

    def test_reject_live_wrong_season_and_incomplete_scores(self):
        for key,value in [('gameStatus',2),('gameId','0042600001'),('gameEt','2025-10-20T19:00:00Z')]:
            g=self.game();g[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):completed_rows(g)
        g=self.game();g['homeTeam']['score']=99
        with self.assertRaises(ValueError):completed_rows(g)

    def test_neutral_home_and_historical_starters(self):
        g=self.game();g['isNeutral']=True
        players,_,_=completed_rows(g)
        self.assertTrue(all(r['is_home']==0 for r in players))
        g=self.game();g['homeTeam']['players'][0]['position']=''
        with self.assertRaises(ValueError):completed_rows(g)

    def test_official_iso_duration_minutes(self):
        g=self.game()
        g['homeTeam']['players'][0]['statistics']['minutes']='PT23M56.00S'
        g['homeTeam']['players'][5]['statistics']['minutes']='PT00M00.00S'
        players,_,excluded=completed_rows(g)
        self.assertAlmostEqual(players[0]['MIN'],23+56/60)
        self.assertEqual(len(excluded),2)
        g['homeTeam']['players'][0]['statistics']['minutes']='PT23M70S'
        with self.assertRaises(ValueError):completed_rows(g)

    def test_explicit_starter_flag_wins_over_position(self):
        g=self.game()
        for p in g['homeTeam']['players']:
            p['starter']=str(int(p['personId']<105))
            p['position']='G'
        g['homeTeam']['players'][5]['statistics']['minutes']='05:00'
        players,_,_=completed_rows(g)
        self.assertEqual(sum(r['is_starter'] for r in players if r['TEAM']=='BOS'),5)
        self.assertEqual(next(r for r in players if r['nba_player_id']==105)['is_starter'],0)

    def test_pending_schedule_never_uses_results_or_confirms_lineup(self):
        c=dict(gameId='0022600001',seasonYear='2026-27',seasonType='Regular Season',gameStatus=1,
               gameTimeEastern='2026-10-20T19:00:00Z',gameTimeUtc='2026-10-20T23:00:00Z',
               homeTeam=dict(teamTricode='BOS'),awayTeam=dict(teamTricode='NYK'))
        rows=schedule_rows(c)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['scheduled_start_utc'],'2026-10-20T23:00:00Z')
        self.assertNotIn('PTS',rows[0])
        self.assertFalse(rows[0]['pregame_verified'])
        other=copy.deepcopy(c);other['gameId']='0062600001'
        self.assertEqual(schedule_rows(other),[])

    def test_profiles_use_only_previous_season_positive_minutes(self):
        rows=[dict(PLAYER='Test',GAMEDATE='2025-10-21',MIN='20',PTS='10'),
              dict(PLAYER='Test',GAMEDATE='2026-04-12',MIN='30',PTS='20'),
              dict(PLAYER='Other',GAMEDATE='2026-04-12',MIN='0',PTS='0')]
        profiles=prior_profiles(rows)
        self.assertEqual(len(profiles),1)
        self.assertEqual(profiles[0]['prev_season_pts_mean'],15)
        self.assertEqual(profiles[0]['prev_season_pts_per_min'],.6)
        rows.append(dict(PLAYER='Test',GAMEDATE='2026-10-20',MIN='20',PTS='99'))
        with self.assertRaises(ValueError):prior_profiles(rows)

    def test_conference_parser_requires_six_complete_divisions(self):
        divisions=['Atlantic','Central','Southeast','Northwest','Pacific','Southwest']
        html=''.join(f'<div class="TeamDivisions_divisionName__x">{d}</div>'+''.join(
            f'<a data-content="{chr(65+i)}A{chr(65+j)}">Schedule</a>' for j in range(5))
            for i,d in enumerate(divisions))
        teams=conference_teams(html)
        self.assertEqual(len(teams),30)
        self.assertEqual(teams['AAA'],'East')
        self.assertEqual(teams['FAA'],'West')
        with self.assertRaises(ValueError):conference_teams(html.replace('Atlantic','Unknown'))


class BoxPageRetryTest(unittest.TestCase):
    def fetcher(self,responses):
        calls=[]
        def fetch(url):
            calls.append(url);r=responses[len(calls)-1]
            if isinstance(r,Exception):raise r
            return r,dict(source_url=url)
        return fetch,calls

    def test_retries_page_without_game_then_records_attempts(self):
        fetch,calls=self.fetcher([{},{'game':None},{'game':{'gameId':'0022600197'}}])
        waits=[]
        g,e=box_page('0022600197',fetch=fetch,sleep=waits.append,waits=(1,2,3))
        self.assertEqual(g['gameId'],'0022600197')
        self.assertEqual(e['box_page_attempts'],3)
        self.assertEqual(waits,[1,2])
        self.assertEqual(calls,['https://www.nba.com/game/0022600197/box-score']*3)

    def test_first_good_page_keeps_evidence_unchanged(self):
        fetch,_=self.fetcher([{'game':{'gameId':'1'}}])
        _,e=box_page('1',fetch=fetch,sleep=self.fail)
        self.assertNotIn('box_page_attempts',e)

    def test_persistent_missing_game_still_fails_with_url(self):
        fetch,calls=self.fetcher([{}]*4)
        with self.assertRaisesRegex(ValueError,r'no game data after 4 tries \(https://www\.nba\.com/game/9/box-score\)'):
            box_page('9',fetch=fetch,sleep=lambda s:None,waits=(1,1,1))
        self.assertEqual(len(calls),4)

    def test_fetch_errors_are_not_swallowed(self):
        fetch,calls=self.fetcher([OSError('timeout')])
        with self.assertRaises(OSError):box_page('9',fetch=fetch,sleep=self.fail)
        self.assertEqual(len(calls),1)

    def test_describe_error_names_missing_field(self):
        self.assertEqual(describe_error(KeyError('game')),"KeyError: missing field 'game'")
        self.assertEqual(describe_error(ValueError('Game clock is not zero')),'ValueError: Game clock is not zero')


class ActionsReportTest(unittest.TestCase):
    def test_annotation_escapes_workflow_command_characters(self):
        self.assertEqual(annotation('Box score 1, a:b','50% done\nnext'),
                         '::error title=Box score 1%2C a%3Ab::50%25 done%0Anext')

    def test_silent_outside_actions(self):
        out=io.StringIO()
        self.assertFalse(report_error('t','m',env={},stream=out))
        self.assertEqual(out.getvalue(),'')

    def test_actions_annotation_and_step_summary(self):
        with tempfile.TemporaryDirectory() as d:
            summary=Path(d)/'summary.md';out=io.StringIO()
            self.assertTrue(report_error('Box score 0022600197',"KeyError: missing field 'game'",
                env={'GITHUB_ACTIONS':'true','GITHUB_STEP_SUMMARY':str(summary)},stream=out))
            self.assertEqual(out.getvalue(),"::error title=Box score 0022600197::KeyError: missing field 'game'\n")
            self.assertIn('0022600197',summary.read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
