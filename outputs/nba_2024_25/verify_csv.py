import csv
import json
from pathlib import Path
from collections import Counter
import unicodedata

root = Path(__file__).parent
raw = json.loads((root / 'source_rows.json').read_text(encoding='utf-8'))
out = root / 'nba_2024_25_regular_season_player_game_logs.csv'
with out.open(encoding='utf-8-sig', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)
assert header == ['PLAYER','TEAM','MATCHUP','GAMEDATE','MIN','PTS','REB','AST']
assert len(rows) == 26306
assert all(len(r) == 8 for r in rows)
def canonical(r):
    return tuple(r[:4]) + tuple(float(x) if x != '-' else '-' for x in r[4:])
expected = []
for r in raw:
    vals = [r['values'][i] for i in [0,1,2,3,5,6,18,19]]
    m,d,y = vals[3].split('/')
    vals[3] = f'{y}-{m}-{d}'
    expected.append(canonical(vals))
assert Counter(map(canonical, rows)) == Counter(expected), 'Saved CSV differs from source'
assert len({(r[0],r[2],r[3]) for r in rows}) == len(rows)
def az(name):
    return ''.join(c for c in unicodedata.normalize('NFKD', name) if not unicodedata.combining(c) and c.isalnum()).casefold()
assert all(az(a[0]) <= az(b[0]) for a,b in zip(rows,rows[1:])), 'Names not A-Z'
assert all(a[3] <= b[3] for a,b in zip(rows,rows[1:]) if a[0] == b[0]), 'Dates not ascending'
print(json.dumps({'verified':True,'rows':len(rows),'players':len({r[0] for r in rows}),'columns':header,'bytes':out.stat().st_size,'source_match':'all rows match'},ensure_ascii=False))
