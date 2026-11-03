"""Manual 2026-27 NBA collection: compact history, upcoming games, source evidence.

Does not fit models, backfill pregame confirmations, or alter frozen seasons.
Run: .venv/Scripts/python.exe -X utf8 -m scripts.fetch_current_season
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path

from scripts.fetch_historical_seasons import page, minutes_seconds, dump
from scripts.merge_game_rosters import ALIASES

ROOT=Path(__file__).resolve().parents[1]
SEASON='2026-27'
# Verified against the official personId in the saved 2025-26 source rows.
CURRENT_NAME_KEYS={1630619:'Moussa Cisse'}


def regular(c):
    return (str(c.get('gameId','')).startswith('00226') and
            c.get('seasonYear')==SEASON and c.get('seasonType')=='Regular Season')


def schedule_rows(c):
    if not regular(c):return []
    day=c['gameTimeEastern'][:10]
    if not '2026-07-01'<=day<'2027-07-01':raise ValueError('Wrong schedule season')
    start=c.get('gameTimeUtc')
    if not start or datetime.fromisoformat(start.replace('Z','+00:00')).utcoffset()!=timedelta(0):
        raise ValueError('Missing or invalid official UTC start')
    home=c['homeTeam']['teamTricode'];away=c['awayTeam']['teamTricode']
    if home==away:raise ValueError('Same teams')
    return [dict(season=SEASON,nba_game_id=c['gameId'],GAMEDATE=day,team_id=t,
                 opponent_id=away if t==home else home,
                 MATCHUP=f'{home} VS. {away}' if t==home else f'{away} @ {home}',
                 game_id=(f'{home} VS. {away}' if t==home else f'{away} @ {home}')+'|'+day,
                 designated_home_team=home,scheduled_start_utc=start,
                 is_neutral_site=c.get('isNeutral'),game_status=c['gameStatus'],
                 pregame_verified=False,lineup_status='not_collected') for t in [home,away]]


def stat(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or int(value)!=value:
        raise ValueError('Invalid count statistic')
    return int(value)


def played_seconds(value):
    if isinstance(value,str) and value.startswith('PT'):
        m=re.fullmatch(r'PT(\d+)M(\d+(?:\.\d+)?)S',value)
        if not m or float(m[2])>=60:raise ValueError('Invalid official ISO minutes')
        return int(m[1])*60+float(m[2])
    return minutes_seconds(value)


def completed_rows(g):
    gid=g['gameId'];day=g['gameEt'][:10]
    if not gid.startswith('00226') or g['gameStatus']!=3 or not '2026-07-01'<=day<'2027-07-01':
        raise ValueError('Not a completed 2026-27 regular-season game')
    neutral=g.get('isNeutral')
    if not isinstance(neutral,bool):raise ValueError('Neutral venue not verified')
    home=g['homeTeam']['teamTricode'];away=g['awayTeam']['teamTricode']
    if home==away:raise ValueError('Same teams')
    players=[];teams=[];excluded=[];seen=set()
    for side,other in [('homeTeam','awayTeam'),('awayTeam','homeTeam')]:
        t=g[side];opp=g[other];code=t['teamTricode']
        matchup=f'{home} VS. {away}' if side=='homeTeam' else f'{away} @ {home}'
        total=0;starters=0
        for p in t['players']:
            pid=int(p['personId'])
            if pid in seen:raise ValueError('Duplicate player')
            seen.add(pid)
            name=(p['firstName']+' '+p['familyName']).strip()
            canonical=CURRENT_NAME_KEYS.get(pid,ALIASES.get(name,name))
            s=p.get('statistics') or {};seconds=played_seconds(s.get('minutes'))
            if not seconds:
                if s.get('points') not in (None,0):raise ValueError('Points without minutes')
                excluded.append(dict(nba_game_id=gid,nba_player_id=pid,PLAYER=canonical,TEAM=code,
                                     GAMEDATE=day,reason='not_positive_minutes'))
                continue
            pts=stat(s['points']);reb=stat(s['reboundsTotal']);ast=stat(s['assists'])
            if 'starter' in p:
                if p['starter'] not in ('0','1',0,1):raise ValueError('Invalid starter flag')
                starter=int(p['starter'])
            else:starter=int(bool(p.get('position')))
            starters+=starter;total+=pts
            players.append(dict(season=SEASON,PLAYER=canonical,TEAM=code,MATCHUP=matchup,GAMEDATE=day,
                MIN=seconds/60,PTS=pts,REB=reb,AST=ast,player_id=canonical,team_id=code,
                game_id=matchup+'|'+day,nba_player_id=pid,nba_game_id=gid,source_player_name=name,
                is_home=int(side=='homeTeam' and not neutral),is_neutral_site=int(neutral),
                is_starter=starter,designated_home_team=home,minutes_source=s['minutes'],
                lineup_source_kind='historical_postgame',pregame_verified=False))
        if starters!=5 or total!=stat(t['score']):raise ValueError('Incomplete starters or score mismatch')
        teams.append(dict(season=SEASON,team_id=code,opponent_id=opp['teamTricode'],GAMEDATE=day,
            game_id=matchup+'|'+day,nba_game_id=gid,points_for=total,points_allowed=stat(opp['score']),
            combined_points=total+stat(opp['score']),status='completed'))
    return players,teams,excluded


def validate_final_box(g,now):
    """Historical final evidence, never a pregame-confirmation shortcut."""
    text=g.get('gameStatusText','').strip()
    if not re.fullmatch(r'Final(?:/(?:\d*OT))?',text):raise ValueError('No explicit Final label')
    period=g.get('period')
    if isinstance(period,bool) or not isinstance(period,int) or period<4:raise ValueError('Incomplete regulation periods')
    if played_seconds(g.get('gameClock'))!=0:raise ValueError('Game clock is not zero')
    start=datetime.fromisoformat(g['gameTimeUTC'].replace('Z','+00:00'))
    if start.utcoffset() is None or now.utcoffset() is None or now-start<timedelta(minutes=48+5*(period-4)):
        raise ValueError('Final before minimum elapsed regulation time')
    expected=(48+5*(period-4))*60*5
    for side in ['homeTeam','awayTeam']:
        seconds=[played_seconds((p.get('statistics') or {}).get('minutes')) or 0 for p in g[side]['players']]
        # Public player minutes can be truncated to whole seconds (observed 3s
        # team drift). Bound the accumulated error by 1s per appeared player.
        tolerance=max(2,sum(s>0 for s in seconds))
        total_minutes=(g[side].get('statistics') or {}).get('minutes')
        if total_minutes is not None and played_seconds(total_minutes)!=expected:raise ValueError('Official team minutes disagree')
        if not math.isclose(sum(seconds),expected,rel_tol=0,abs_tol=tolerance):raise ValueError('Final team minutes incomplete')
    completed_rows(g)


def resolve_final_status(g,reference,now):
    """Return a derived historical Final only with a second official Final feed.

    Raw source objects are preserved. A numeric 1/2 alone never means completed.
    """
    if g['gameStatus'] not in (1,2) or reference['gameStatus']!=3 or reference.get('gameStatusText')!='Final' or reference['gameId']!=g['gameId']:
        raise ValueError('Independent official Final reference missing')
    seen=set()
    for side in ['homeTeam','awayTeam']:
        team=g[side];ref=reference[side]
        if team['teamId']!=ref['teamId'] or team['teamTricode']!=ref['teamAbbreviation']:raise ValueError('Final reference team mismatch')
        expected=set()
        for p in team['players']:
            starter=int(p['starter']) if 'starter' in p else int(bool(p.get('position')))
            if starter:expected.add(int(p['personId']))
        starters=set()
        for p in ref['players']:
            pid=int(p['personId'])
            if pid in seen or p['teamId']!=team['teamId']:raise ValueError('Final reference player identity mismatch')
            seen.add(pid)
            if p.get('position'):
                if p['lineupStatus']!='Confirmed' or p['rosterStatus']!='Active':raise ValueError('Final reference starters unconfirmed')
                starters.add(pid)
        if len(starters)!=5 or starters!=expected:raise ValueError('Final reference starting five mismatch')
    resolved=dict(g,gameStatus=3)
    validate_final_box(resolved,now)
    return resolved


def reconcile_overdue_cards(cards,now,out):
    """Inspect past scheduled non-final cards; preserve genuine pending games."""
    resolved={};attempts=[]
    for gid,c in sorted(cards.items()):
        if c['gameStatus']==3:continue
        start=datetime.fromisoformat(c['gameTimeUtc'].replace('Z','+00:00'))
        if start.utcoffset() is None:raise ValueError('Schedule UTC missing')
        if start>=now:continue
        attempt=dict(nba_game_id=gid,daily_numeric_status=c['gameStatus'],resolved=False)
        try:
            props,box_evidence=page(f'https://www.nba.com/game/{gid}/box-score');g=props['game']
            if g['gameId']!=gid or g['gameEt'][:10]!=c['gameTimeEastern'][:10]:raise ValueError('Reconciliation date/ID mismatch')
            for side in ['homeTeam','awayTeam']:
                if g[side]['teamId']!=c[side]['teamId'] or g[side]['teamTricode']!=c[side]['teamTricode']:
                    raise ValueError('Reconciliation team mismatch')
            attempt.update(box_numeric_status=g['gameStatus'],box_status_text=g.get('gameStatusText'),box_evidence=box_evidence)
            if g['gameStatus']==3:
                validate_final_box(g,now);fixed=g;attempt['method']='official_box_final'
            elif re.fullmatch(r'Final(?:/(?:\d*OT))?',g.get('gameStatusText','').strip()):
                day=g['gameEt'][:10];url='https://stats.nba.com/js/data/leaders/00_daily_lineups_'+day.replace('-','')+'.json'
                raw=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Referer':'https://www.nba.com/'}),timeout=15).read()
                refs=[x for x in json.loads(raw)['games'] if x['gameId']==gid]
                if len(refs)!=1:raise ValueError('Independent Final game not unique')
                ref=refs[0];fixed=resolve_final_status(g,ref,now)
                attempt.update(method='official_final_text_full_box_and_independent_final_feed',reference_numeric_status=ref['gameStatus'],
                    reference_url=url,reference_observed_at_utc=datetime.now(timezone.utc).isoformat(),reference_sha256=hashlib.sha256(raw).hexdigest())
                dump(out/'status_evidence'/f'{gid}_final_reference.json',dict(game=ref,source_url=url,sha256=attempt['reference_sha256'],fetched_at_utc=attempt['reference_observed_at_utc']))
            else:
                attempt['reason']='Source still not final';attempts.append(attempt);continue
            # Only the derived card is adjusted; both original status codes remain in the audit.
            derived=dict(c,gameStatus=3)
            for side in ['homeTeam','awayTeam']:derived[side]=dict(c[side],score=fixed[side]['score'])
            attempt['resolved']=True
            resolved[gid]=dict(card=derived,game=fixed,box_evidence=box_evidence,resolution=attempt)
        except (ValueError,KeyError,TypeError,OSError) as exc:attempt['reason']=str(exc)
        attempts.append(attempt)
    dump(out/'status_resolution.json',attempts)
    return resolved


def write_csv(path,rows,empty_fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else empty_fields)
        w.writeheader();w.writerows(rows)


def prior_profiles(rows):
    totals=defaultdict(lambda:[0,0.,0.]);seen=set()
    for r in rows:
        d=r['GAMEDATE'];date.fromisoformat(d)
        if not '2025-10-21'<=d<='2026-04-12':raise ValueError('Wrong prior season')
        name=ALIASES.get(r['PLAYER'],r['PLAYER']);key=name,d
        if key in seen:raise ValueError('Duplicate prior player date')
        seen.add(key);minutes=float(r['MIN']);points=stat(float(r['PTS']))
        if not math.isfinite(minutes) or minutes<0:raise ValueError('Invalid prior minutes')
        if minutes==0:
            if points:raise ValueError('Prior points without minutes')
            continue
        t=totals[name];t[0]+=1;t[1]+=points;t[2]+=minutes
    return [dict(player_id=name,prev_season_pts_mean=p/n,prev_season_min_mean=m/n,
                 prev_season_pts_per_min=p/m,prev_season_games=n,has_prev_season=1)
            for name,(n,p,m) in sorted(totals.items())]


def conference_teams(html):
    groups=re.split(r'TeamDivisions_divisionName[^>]*>([^<]+)</div>',html)
    expected={'Atlantic':'East','Central':'East','Southeast':'East','Northwest':'West','Pacific':'West','Southwest':'West'}
    teams={};found=set()
    for division,body in zip(groups[1::2],groups[2::2]):
        if division not in expected or division in found:raise ValueError('Unknown or repeated division')
        codes=set(re.findall(r'data-content="([A-Z]{3})"',body))
        if len(codes)!=5 or set(teams)&codes:raise ValueError('Incomplete division or duplicate team')
        found.add(division);teams.update({code:expected[division] for code in codes})
    if found!=set(expected) or len(teams)!=30:raise ValueError('Incomplete conference registry')
    return dict(sorted(teams.items()))


def collect_conferences(out):
    url='https://www.nba.com/teams'
    raw=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=20).read()
    registry=dict(season=SEASON,teams=conference_teams(raw.decode()),source_url=url,
        conference_definition_source='https://www.nba.com/news/about',
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),html_sha256=hashlib.sha256(raw).hexdigest(),
        note='Current official team division listing; conference grouping per NBA About. Collection time is not publication time.')
    dump(out/'conferences_2026_27.json',registry)
    return registry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through',default=datetime.now(timezone(timedelta(hours=8))).date().isoformat())
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();last=date.fromisoformat(args.through)
    today=datetime.now(timezone(timedelta(hours=8))).date()
    if not date(2026,10,1)<=last<=min(today,date(2027,6,30)) or not 1<=args.workers<=4:
        raise ValueError('Invalid date or workers (1..4)')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'outputs/nba_2026_27'/stamp;out.mkdir(parents=True,exist_ok=False)
    conferences=collect_conferences(out)
    dates=[(date(2026,10,1)+timedelta(days=i)).isoformat() for i in range((last-date(2026,10,1)).days+1)]
    cards={};daily=[];errors=[]
    def daily_page(d):
        p,e=page('https://www.nba.com/games?date='+d)
        if p.get('selectedDate')!=d:raise ValueError('Daily page date mismatch')
        cs=[c.get('cardData',{}) for m in p['gameCardFeed']['modules'] for c in m.get('cards',[])]
        return d,[c for c in cs if regular(c)],e
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for d,cs,e in pool.map(daily_page,dates):
            daily.append(dict(day=d,game_ids=[c['gameId'] for c in cs],**e))
            for c in cs:
                if c['gameId'] in cards:raise ValueError('Duplicate indexed game')
                if c['gameTimeEastern'][:10]!=d:raise ValueError('Game/date mismatch')
                cards[c['gameId']]=c
    dump(out/'daily_index.json',daily)
    reconciled=reconcile_overdue_cards(cards,datetime.now(timezone.utc),out)
    for gid,record in reconciled.items():cards[gid]=record['card']
    finals={gid:c for gid,c in cards.items() if c['gameStatus']==3}
    pending=[r for c in cards.values() if c['gameStatus']!=3 for r in schedule_rows(c)]
    players=[];teams=[];excluded=[];evidence=[]
    def box(item):
        gid,c=item
        try:
            resolution=reconciled.get(gid)
            if resolution:g=resolution['game'];e=resolution['box_evidence']
            else:
                p,e=page(f'https://www.nba.com/game/{gid}/box-score');g=p['game']
            if g['gameId']!=gid or g['gameEt'][:10]!=c['gameTimeEastern'][:10]:raise ValueError('Box/index mismatch')
            for side in ['homeTeam','awayTeam']:
                if g[side]['teamTricode']!=c[side]['teamTricode'] or stat(g[side]['score'])!=stat(c[side]['score']):
                    raise ValueError('Box/index team or score mismatch')
            validate_final_box(g,datetime.now(timezone.utc))
            rows,t,ex=completed_rows(g)
            resolution_info=resolution['resolution'] if resolution else None
            dump(out/'evidence'/f'{gid}.json',dict(players=rows,teams=t,excluded=ex,status_resolution=resolution_info,**e))
            return rows,t,ex,dict(nba_game_id=gid,status_resolution=resolution_info,**e),None
        except Exception as exc:return [],[],[],None,dict(nba_game_id=gid,error=str(exc))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,(ps,ts,ex,ev,err) in enumerate(pool.map(box,sorted(finals.items())),1):
            players.extend(ps);teams.extend(ts);excluded.extend(ex)
            if ev:evidence.append(ev)
            if err:errors.append(err)
            if i%20==0 or i==len(finals):print(json.dumps(dict(processed=i,total=len(finals),errors=len(errors))),flush=True)
    players.sort(key=lambda r:(r['GAMEDATE'],r['nba_game_id'],r['TEAM'],r['PLAYER']))
    teams.sort(key=lambda r:(r['GAMEDATE'],r['nba_game_id'],r['team_id']))
    if not {r['team_id'] for r in teams+pending}<=set(conferences['teams']):raise ValueError('Unknown conference team')
    for rows in [players,teams]:
        keys=[(r.get('player_id',r.get('team_id')),r['GAMEDATE']) for r in rows]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate identity/date')
    # Audit metadata live outside model tables; collection time is not publication time.
    write_csv(out/'player_games.csv',players,['season','PLAYER','TEAM','MATCHUP','GAMEDATE','MIN','PTS','REB','AST'])
    write_csv(out/'team_games.csv',teams,['season','team_id','GAMEDATE','status'])
    write_csv(out/'pending_schedule.csv',pending,['season','nba_game_id','GAMEDATE','team_id','scheduled_start_utc','pregame_verified'])
    write_csv(out/'excluded_nonparticipants.csv',excluded,['nba_game_id','nba_player_id','PLAYER','TEAM','GAMEDATE','reason'])
    prior_source=ROOT/'outputs/week1/model_data/nba_2025_26_player_games_merged.csv'
    with prior_source.open(encoding='utf-8-sig',newline='') as f:profiles=prior_profiles(list(csv.DictReader(f)))
    write_csv(out/'previous_season_profiles.csv',profiles,['player_id'])
    dump(out/'sources.json',evidence)
    summary=dict(season=SEASON,through=last.isoformat(),collected_at_utc=datetime.now(timezone.utc).isoformat(),
        completed_games_indexed=len(finals),completed_games_collected=len(evidence),player_rows=len(players),team_rows=len(teams),
        excluded_rows=len(excluded),pending_games=len(pending)//2,errors=errors,complete=not errors,
        reconciled_games=sorted(reconciled),
        first_game_date=min((r['GAMEDATE'] for r in players),default=None),last_game_date=max((r['GAMEDATE'] for r in players),default=None),
        source_publication_time=None,pregame_lineups_collected=False,prediction_ready=False,
        previous_season_profiles=len(profiles),previous_season_source=str(prior_source.relative_to(ROOT)),
        previous_season_source_sha256=hashlib.sha256(prior_source.read_bytes()).hexdigest(),
        file_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))})
    dump(out/'manifest.json',summary)
    if not errors:
        dump(out.parent/'latest_success.json',dict(snapshot=stamp,manifest=str((out/'manifest.json').relative_to(ROOT))))
    print(json.dumps(dict(directory=str(out),**summary),ensure_ascii=False,indent=2),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
