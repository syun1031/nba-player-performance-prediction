import csv
import json
import re
from pathlib import Path
from collections import defaultdict

root=Path(__file__).parent
raw=json.loads((root/'source_rows.json').read_text(encoding='utf-8'))
boxes=json.loads((root/'roster_boxes.json').read_text(encoding='utf-8'))
games=defaultdict(list)
for r in raw: games[r['links'][2].split('/')[-1]].append(r)
expected={}
for b in boxes:
 source=games[b['gid']]
 away=next(r for r in source if ' @ ' in r['values'][2])
 matchup=away['values'][2]
 m,d,y=away['values'][3].split('/')
 date=f'{y}-{m}-{d}'
 mapping={r['links'][0].split('/')[-1]:r['values'][1] for r in source}
 for players in b['teams']:
  active=[]
  for p in players:
   mm=re.fullmatch(r'(\d+):(\d{2})',p.get('min',''))
   if mm and int(mm[1])*60+int(mm[2])>0: active.append(p)
  team={mapping[p['pid']] for p in active}
  assert len(team)==1
  expected[(date,matchup,team.pop())]=[(('*' if p['position'] else '')+p['name'].strip()) for p in sorted(active,key=lambda p:bool(p['position']),reverse=True)]
path=root/'nba_2025_26_regular_season_game_rosters.csv'
with path.open(encoding='utf-8-sig',newline='') as f:
 reader=csv.reader(f)
 header=next(reader)
 rows=list(reader)
assert len(rows)==2460
assert header==['MATCHUP','GAMEDATE','TEAM']+[f'PLAYER_{i}' for i in range(1,16)]
seen=set()
for row in rows:
 matchup,date,team=row[:3]
 players=[p for p in row[3:] if p]
 key=(date,matchup,team)
 assert key not in seen
 seen.add(key)
 assert players==expected[key]
 assert len([p for p in players if p.startswith('*')])==5
 assert len({p.lstrip('*') for p in players})==len(players)
 assert all(p.startswith('*') for p in players[:5])
 assert not any('DNP' in p for p in players)
for away,home in zip(rows[::2],rows[1::2]):
 assert away[:2]==home[:2]
 visiting,hosting=away[0].split(' @ ')
 assert away[2]==visiting
 assert home[2]==hosting
keys=[(r[1],r[0]) for r in rows]
assert keys==sorted(keys)
assert seen==set(expected)
assert sum(bool(p) for r in rows for p in r[3:])==26648
print(json.dumps({'verified':True,'games':len(rows)//2,'team_rows':len(rows),'player_entries':26648,'starters_per_team':5,'guest_then_home':True,'all_rows_match_official_source':True,'bytes':path.stat().st_size}))
