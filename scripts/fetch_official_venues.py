"""Read public NBA game pages and retain compact venue evidence per saved game."""
import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/week1/venue_audit'


def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Accept-Encoding':'identity'})
    return urllib.request.urlopen(req,timeout=20).read()


def extract_game(payload):
    text=payload.decode('utf-8')
    match=re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',text,re.S)
    if not match:
        raise ValueError('No official page data')
    return json.loads(match.group(1))['props']['pageProps']['game']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/'games'
    cache.mkdir(exist_ok=True)
    rows=json.loads((ROOT/'outputs/nba_2025_26/source_rows.json').read_text(encoding='utf-8'))
    games=defaultdict(list)
    for row in rows:
        games[row['links'][2].rsplit('/',1)[-1]].append(row)
    todo=sorted(games)
    if args.limit:
        todo=todo[:args.limit]
    def one(gid):
        dest=cache/(gid+'.json')
        if dest.exists():
            record=json.loads(dest.read_text(encoding='utf-8'))
            assert record['gameId']==gid
            return gid,record,None
        source=games[gid]
        teams=sorted({r['values'][1] for r in source})
        url=f'https://www.nba.com/game/{teams[0].lower()}-vs-{teams[1].lower()}-{gid}'
        try:
            payload=fetch(url)
            g=extract_game(payload)
            if g['gameId']!=gid or not isinstance(g['isNeutral'],bool):
                raise ValueError('Game identity or neutral type mismatch')
            actual_teams=sorted([g['homeTeam']['teamTricode'],g['awayTeam']['teamTricode']])
            if actual_teams!=teams:
                raise ValueError('Team mismatch')
            mm,dd,yyyy=source[0]['values'][3].split('/')
            expected=f'{yyyy}-{mm}-{dd}'
            game_et=g.get('gameEt','')
            if game_et[:10]!=expected:
                raise ValueError(f'Date mismatch {expected} {game_et}')
            if not g.get('arena',{}).get('arenaName'):
                raise ValueError('No arena name')
            record=dict(gameId=gid,GAMEDATE=expected,teams=actual_teams,
                        isNeutral=g['isNeutral'],arena=g['arena'],
                        homeTeam=g['homeTeam']['teamTricode'],awayTeam=g['awayTeam']['teamTricode'],
                        gameEt=game_et,gameTimeUTC=g.get('gameTimeUTC'),
                        source_url=url,fetched_at_utc=datetime.now(timezone.utc).isoformat(),
                        html_sha256=hashlib.sha256(payload).hexdigest())
            dest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            time.sleep(0.15)
            return gid,record,None
        except Exception as exc:
            return gid,None,dict(gameId=gid,url=url,error=f'{type(exc).__name__}: {exc}')
    successes,errors=[],[]
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,(gid,record,error) in enumerate(pool.map(one,todo),1):
            if error:
                errors.append(error)
            else:
                successes.append(record)
            if i%50==0 or i==len(todo):
                print(json.dumps(dict(processed=i,total=len(todo),success=len(successes),
                                      errors=len(errors),elapsed_seconds=round(time.monotonic()-start)),ensure_ascii=False),flush=True)
    summary=dict(expected_games=len(games),attempted_games=len(todo),success=len(successes),
                 errors=errors,neutral_games=[r for r in successes if r['isNeutral']],
                 complete=len(successes)==len(games) and not errors)
    (OUT/'fetch_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if errors:
        print(json.dumps(errors[:5],ensure_ascii=False),flush=True)
        raise SystemExit(1)


if __name__=='__main__':
    main()
