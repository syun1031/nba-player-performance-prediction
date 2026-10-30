"""Resumable public NBA page collection, isolated from original season CSVs."""
import argparse
import concurrent.futures
from datetime import date,timedelta,datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
from scripts.build_features_baseline import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/season_extension_2024_25'
RANGES={'2023-24':('2023-10-24','2024-04-14'),'2024-25':('2024-10-22','2025-04-13')}


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8');temp.replace(path)


def minutes_seconds(value):
    if value is None or value in ('','--') or str(value).upper().startswith(('DNP','DND','NWT','DID NOT')): return None
    m=re.fullmatch(r'(\d+):(\d{2})',str(value))
    # NBA embedded pages occasionally render rounded seconds as 13:60.
    # Carry 60 into the next minute; retain the original string as evidence.
    if m is None or int(m[2])>60: raise ValueError('Unexpected minute value: '+str(value))
    return int(m[1])*60+int(m[2])


def page(url):
    last=None
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Accept-Encoding':'identity'})
            payload=urllib.request.urlopen(req,timeout=20).read()
            m=re.search(rb'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',payload,re.S)
            if not m: raise ValueError('No official embedded page data')
            props=json.loads(m.group(1))['props']['pageProps']
            time.sleep(.10)
            return props,dict(source_url=url,fetched_at_utc=datetime.now(timezone.utc).isoformat(),html_sha256=hashlib.sha256(payload).hexdigest())
        except Exception as e:
            last=e
            if attempt<2:time.sleep(attempt+1)
    raise last


def regular_cards(props,day,season):
    if props.get('selectedDate')!=day:raise ValueError('Daily page date mismatch')
    result=[]; prefix='002'+season[2:4]
    for module in props['gameCardFeed']['modules']:
        for card in module.get('cards',[]):
            gid=card.get('cardData',{}).get('gameId','')
            if gid.startswith(prefix): result.append(gid)
    if len(set(result))!=len(result):raise ValueError('Duplicate daily games')
    return sorted(result)


def compact_game(g,gid,season):
    start,end=RANGES[season]
    if g['gameId']!=gid or not gid.startswith('002'+season[2:4]) or g['gameStatus']!=3 or not start<=g['gameEt'][:10]<=end or not isinstance(g['isNeutral'],bool):
        raise ValueError('Unexpected game identity, season, status or venue')
    result={k:g.get(k) for k in ['gameId','gameEt','gameTimeUTC','isNeutral','arena','gameCode']}
    ids=set()
    for side in ['homeTeam','awayTeam']:
        team=g[side]; players=[]
        for p in team['players']:
            stats=p.get('statistics') or {}; seconds=minutes_seconds(stats.get('minutes'))
            pid=int(p['personId'])
            if pid in ids:raise ValueError('Duplicated player ID')
            ids.add(pid)
            players.append(dict(personId=pid,name=(p['firstName']+' '+p['familyName']).strip(),position=p.get('position',''),
                seconds=seconds,minutes=stats.get('minutes'),points=stats.get('points'),rebounds=stats.get('reboundsTotal'),assists=stats.get('assists')))
        if sum(bool(p['position']) for p in players)!=5 or any(p['position'] and not p['seconds'] for p in players):
            raise ValueError('Starting five not verified')
        if sum(p['points'] for p in players if p['seconds'])!=team['score']:raise ValueError('Player sum differs from team score')
        result[side]=dict(teamTricode=team['teamTricode'],teamId=team['teamId'],score=team['score'],players=players)
    if result['homeTeam']['teamTricode']==result['awayTeam']['teamTricode']:raise ValueError('Same two teams')
    return result


def legacy_game(gid,season,day):
    """Official NBA legacy feed plus dated game card when the game page fails."""
    url=f'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{season[:4]}/scores/gamedetail/{gid}_gamedetail.json'
    request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Referer':'https://www.nba.com/'})
    data=urllib.request.urlopen(request,timeout=20).read();g=json.loads(data)['g']
    props,evidence=page('https://www.nba.com/games?date='+day)
    cards=[c['cardData'] for m in props['gameCardFeed']['modules'] for c in m.get('cards',[]) if c.get('cardData',{}).get('gameId')==gid]
    if len(cards)!=1 or g['gid']!=gid or g['gdte']!=day:raise ValueError('Legacy game/date mismatch')
    card=cards[0]
    result=dict(gameId=gid,gameEt=g['etm'],gameTimeUTC=None,gameStatus=g['st'],isNeutral=card['isNeutral'],
                arena=dict(arenaName=g['an'],arenaCity=g['ac']),gameCode=g['gcode'])
    for old,side in [('hls','homeTeam'),('vls','awayTeam')]:
        t=g[old];players=[]
        for p in t['pstsg']:
            seconds=p['totsec'];minutes=f'{seconds//60}:{seconds%60:02}'
            if not seconds and p.get('memo'):minutes=None
            players.append(dict(personId=p['pid'],firstName=p['fn'],familyName=p['ln'],position=p['pos'],
                statistics=dict(minutes=minutes,points=p['pts'],reboundsTotal=p['reb'],assists=p['ast'])))
        result[side]=dict(teamTricode=t['ta'],teamId=t['tid'],score=t['s'],players=players)
    return dict(game=compact_game(result,gid,season),source_url=url,fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        json_sha256=hashlib.sha256(data).hexdigest(),venue_evidence=evidence,source_kind='official_legacy_feed_and_daily_card')


def collect_index(workers):
    season='2023-24'; start,end=RANGES[season]; dates={}
    for anchor in [start,'2024-01-01']:
        cache=OUT/'cache'/('calendar_'+anchor[:4]+'.json')
        if cache.exists():calendar=json.loads(cache.read_text(encoding='utf-8'))
        else:
            props,evidence=page('https://www.nba.com/games?date='+anchor)
            calendar=dict(counts=props['allGamesInCurrentYear'][anchor[:4]],**evidence); dump(cache,calendar)
        dates.update({d:n for d,n in calendar['counts'].items() if start<=d<=end and n})
    def one(day):
        path=OUT/'cache/days'/f'{day}.json'
        if path.exists():result=json.loads(path.read_text(encoding='utf-8'))
        else:
            p,e=page('https://www.nba.com/games?date='+day)
            result=dict(day=day,game_ids=regular_cards(p,day,season),**e);dump(path,result)
        if result['day']!=day:raise ValueError('Cached day mismatch')
        return result
    gids={}; errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(one,d):d for d in sorted(dates)}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            d=futures[f]
            try:
                result=f.result()
                for gid in result['game_ids']:
                    if gid in gids:raise ValueError('Duplicate indexed game')
                    gids[gid]=d
            except Exception as e:errors.append(dict(day=d,error=str(e)))
            if i%25==0 or i==len(dates):print(json.dumps(dict(stage='index_2023',processed=i,total=len(dates),games=len(gids),errors=len(errors))),flush=True)
    dump(OUT/'index_2023_24.json',dict(season=season,games=gids,errors=errors,complete=len(gids)==1230 and not errors))
    if errors or len(gids)!=1230:raise ValueError('Incomplete 2023 index: '+str(errors[:3]))
    return gids


def collect_boxes(season,index,workers):
    errors=[]; ok=0
    def one(gid):
        path=OUT/'cache'/season/'games'/f'{gid}.json'
        if path.exists():result=json.loads(path.read_text(encoding='utf-8'))
        else:
            try:
                p,e=page(f'https://www.nba.com/game/{gid}/box-score')
                result=dict(game=compact_game(p['game'],gid,season),**e)
            except Exception:
                result=legacy_game(gid,season,index[gid])
            dump(path,result)
        if result['game']['gameId']!=gid or result['game']['gameEt'][:10]!=index[gid]:raise ValueError('Game cache/index mismatch')
        return gid
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(one,gid):gid for gid in sorted(index)}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            try:f.result();ok+=1
            except Exception as e:errors.append(dict(gameId=futures[f],error=str(e)))
            if i%50==0 or i==len(index):print(json.dumps(dict(stage=season,processed=i,total=len(index),ok=ok,errors=len(errors))),flush=True)
    result=dict(season=season,expected_games=1230,success=ok,errors=errors,complete=ok==1230 and not errors)
    dump(OUT/f'collection_{season}.json',result)
    if errors or ok!=1230:raise ValueError('Incomplete official boxes: '+str(errors[:5]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if not 1<=args.workers<=6:raise ValueError('Workers must be 1..6')
    original=ROOT/'outputs/nba_2024_25/nba_2024_25_regular_season_player_game_logs.csv'; before=sha(original)
    rows=json.loads((original.parent/'source_rows.json').read_text(encoding='utf-8')); index={}
    for r in rows:
        gid=r['links'][2].rsplit('/',1)[-1]; mm,dd,year=r['values'][3].split('/');day=f'{year}-{mm}-{dd}'
        if gid in index and index[gid]!=day:raise ValueError('Original source game dates differ')
        index[gid]=day
    if len(index)!=1230:raise ValueError('Original game coverage')
    dump(OUT/'index_2024_25.json',dict(season='2024-25',games=index,original_csv_sha256=before))
    previous=collect_index(args.workers)
    for season,gids in [('2024-25',index),('2023-24',previous)]:collect_boxes(season,gids,args.workers)
    if sha(original)!=before:raise ValueError('Original CSV changed')
    print('Official collection complete; original CSV unchanged',flush=True)


if __name__=='__main__':main()
