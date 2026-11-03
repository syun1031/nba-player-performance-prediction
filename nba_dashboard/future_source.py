"""Official NBA pregame receipts. Confirmed postgame data never unlocks forecasts."""
from datetime import datetime,timedelta,timezone,date
import hashlib
import json
import re
import urllib.request
from urllib.error import HTTPError,URLError


class SourceIssue(ValueError):pass


def utc_now():return datetime.now(timezone.utc)


def aware(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.utcoffset() is None:raise ValueError('Missing timezone')
    return parsed.astimezone(timezone.utc)


def valid_day(day):
    d=date.fromisoformat(day)
    if d.isoformat()!=day or not '2026-07-01'<=day<'2027-07-01':raise ValueError('Invalid season date')
    return day


def lineup_url(day):return 'https://stats.nba.com/js/data/leaders/00_daily_lineups_'+valid_day(day).replace('-','')+'.json'


def game_url(gid):
    if not re.fullmatch(r'00226\d{5}',gid):raise ValueError('Invalid official game id')
    return 'https://www.nba.com/game/'+gid+'/box-score'


def receive(url,*,html=False):
    request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Accept-Encoding':'identity','Referer':'https://www.nba.com/'})
    try:
        with urllib.request.urlopen(request,timeout=12) as response:raw=response.read(5_000_001)
        if len(raw)>5_000_000:raise SourceIssue('官網資料過大，暫時無法驗證。')
        if html:
            match=re.search(rb'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',raw,re.S)
            if match is None:raise SourceIssue('官網頁面格式已變更，暫時無法驗證。')
            data=json.loads(match[1])['props']['pageProps']
        else:data=json.loads(raw)
        return dict(data=data,url=url,observed_at_utc=utc_now().isoformat(),sha256=hashlib.sha256(raw).hexdigest())
    except HTTPError as exc:
        if exc.code==404:raise SourceIssue('官網尚未公布這個日期或比賽的資料。') from exc
        raise SourceIssue('官網暫時無法讀取（HTTP '+str(exc.code)+'），請稍後再查詢。') from exc
    except (URLError,TimeoutError,OSError,KeyError,json.JSONDecodeError) as exc:
        raise SourceIssue('無法取得完整官網資料，請稍後再查詢。') from exc


def parse_schedule(props,day):
    valid_day(day)
    if props.get('selectedDate')!=day or props.get('hasError'):raise ValueError('Schedule date mismatch')
    games=[];seen=set()
    for module in props['gameCardFeed']['modules']:
        for card in module.get('cards',[]):
            c=card.get('cardData',{});gid=c.get('gameId','')
            if not gid.startswith('00226'):continue
            game_url(gid)
            if c.get('seasonYear')!='2026-27' or c.get('seasonType')!='Regular Season' or c['gameTimeEastern'][:10]!=day:
                raise ValueError('Schedule season/type/date mismatch')
            if gid in seen:raise ValueError('Duplicate scheduled game')
            seen.add(gid);start=aware(c['gameTimeUtc']).isoformat()
            home=c['homeTeam'];away=c['awayTeam']
            if home['teamId']==away['teamId'] or home['teamTricode']==away['teamTricode']:raise ValueError('Same teams')
            games.append(dict(nba_game_id=gid,GAMEDATE=day,home_team=home['teamTricode'],away_team=away['teamTricode'],
                home_team_id=int(home['teamId']),away_team_id=int(away['teamId']),scheduled_start_utc=start,game_status=c['gameStatus']))
    return sorted(games,key=lambda g:(g['scheduled_start_utc'],g['nba_game_id']))


def query_schedule(day):
    receipt=receive('https://www.nba.com/games?date='+valid_day(day),html=True)
    try:games=parse_schedule(receipt['data'],day)
    except (ValueError,KeyError,TypeError) as exc:raise SourceIssue('官網賽程未通過日期與比賽核對。') from exc
    return dict(games=games,source={k:v for k,v in receipt.items() if k!='data'})


def assess_context(schedule,box_receipt,lineup_receipt,now):
    result=dict(ready=False,reasons=[],teams={},schedule=dict(schedule),sources=[box_receipt,lineup_receipt])
    reasons=result['reasons']
    try:
        if now.utcoffset() is None:raise ValueError('Naive clock')
        gid=schedule['nba_game_id'];day=valid_day(schedule['GAMEDATE']);start=aware(schedule['scheduled_start_utc'])
        if box_receipt['url']!=game_url(gid) or lineup_receipt['url']!=lineup_url(day):raise ValueError('Wrong source URL')
        for receipt in [box_receipt,lineup_receipt]:
            age=(now-aware(receipt['observed_at_utc'])).total_seconds()
            if age<0 or age>120:reasons.append('source_stale')
            if not re.fullmatch(r'[a-f0-9]{64}',receipt['sha256']):raise ValueError('Invalid receipt digest')
        box=box_receipt['data']
        if box['gameId']!=gid or box['gameEt'][:10]!=day or aware(box['gameTimeUTC'])!=start:raise ValueError('Box identity/start mismatch')
        if not isinstance(box['isNeutral'],bool):raise ValueError('Venue unknown')
        result['is_neutral_site']=int(box['isNeutral'])
        matches=[g for g in lineup_receipt['data']['games'] if g['gameId']==gid]
        if len(matches)!=1:reasons.append('lineup_unpublished');return result
        game=matches[0]
        if any(status!=1 for status in [schedule['game_status'],box['gameStatus'],game['gameStatus']]):reasons.append('game_not_pregame')
        if now>=start:reasons.append('game_started')
        elif now<start-timedelta(minutes=30):reasons.append('before_prediction_window')
        all_ids=set()
        for side,label in [('homeTeam','home'),('awayTeam','away')]:
            team=game[side];code=schedule[label+'_team'];tid=schedule[label+'_team_id']
            if team['teamAbbreviation']!=code or int(team['teamId'])!=tid or box[side]['teamTricode']!=code or int(box[side]['teamId'])!=tid:
                raise ValueError('Team identity mismatch')
            people=[];active=[];starters=[]
            if not team['players']:reasons.append('lineup_unpublished')
            for p in team['players']:
                pid=int(p['personId']);name=p['playerName']
                if pid<=0 or pid in all_ids or int(p['teamId'])!=tid or not isinstance(name,str) or not name.strip():
                    raise ValueError('Player identity mismatch')
                all_ids.add(pid)
                if p['lineupStatus']!='Confirmed':reasons.append('lineup_not_confirmed')
                if p['rosterStatus'] not in ['Active','Inactive']:reasons.append('roster_unknown')
                position=p.get('position','')
                if position and position not in ['PG','SG','SF','PF','C','G','F']:raise ValueError('Unknown starter position')
                person=dict(nba_player_id=str(pid),name=name.strip(),position=position,
                            active=p['rosterStatus']=='Active',source_timestamp_raw=p.get('timestamp'))
                people.append(person)
                if person['active']:active.append(str(pid))
                if position:
                    starters.append(str(pid))
                    if not person['active']:reasons.append('inactive_starter')
            if len(starters)!=5:reasons.append('incomplete_starting_five')
            result['teams'][code]=dict(nba_team_id=tid,players=people,starters=starters,active=active)
        result['reasons']=list(dict.fromkeys(reasons));result['ready']=not reasons
    except (KeyError,TypeError,ValueError,OverflowError):
        result['ready']=False;result['reasons']=list(dict.fromkeys(reasons+['source_mismatch']))
    return result


def query_context(schedule):
    # Re-read the daily schedule to catch postponements or a changed start time.
    daily=query_schedule(schedule['GAMEDATE'])
    matches=[g for g in daily['games'] if g['nba_game_id']==schedule['nba_game_id']]
    if len(matches)!=1:raise SourceIssue('官網已找不到這場比賽，請重新查詢賽程。')
    schedule=matches[0]
    box=receive(game_url(schedule['nba_game_id']),html=True)
    try:box['data']=box['data']['game']
    except KeyError as exc:raise SourceIssue('官網比賽資料不完整。') from exc
    lineups=receive(lineup_url(schedule['GAMEDATE']))
    result=assess_context(schedule,box,lineups,utc_now())
    result['schedule_source']=daily['source']
    return result


REASONS={'source_stale':'先發資料已超過兩分鐘，請重新查詢。','lineup_unpublished':'等待官網公布本場名單。',
    'game_not_pregame':'官網顯示比賽已開始或不在待賽狀態。','game_started':'已到開賽時間，停止產生賽前預測。',
    'before_prediction_window':'尚未到開賽前30分鐘，先顯示資料狀態。','lineup_not_confirmed':'名單仍是預估，等待官網正式確認。',
    'roster_unknown':'官網未確認啟用名單。','inactive_starter':'先發與未啟用狀態矛盾，等待更新。',
    'incomplete_starting_five':'尚未確認兩隊各五位先發。','source_mismatch':'官網身份、賽程或資料格式未通過核對。'}
