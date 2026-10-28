import csv
import json
from datetime import datetime
from pathlib import Path

root = Path(__file__).parent
sources = json.loads((root/'source_seasons.json').read_text(encoding='utf-8'))
path = root/'lebron_james_lakers_2018_19_to_2025_26_regular_season.csv'
with path.open(encoding='utf-8-sig',newline='') as f:
    rows = list(csv.reader(f))
assert rows.pop(0) == ['MATCHUP','GAMEDATE','MIN','PTS','FG%','REB','AST','STL','BLK']
expected = []
for s in sources:
    assert len(s['rows']) == int(s['countText'].split()[0])
    for r in s['rows']:
        v=r['values']
        date,matchup=v[0].split(' - ')
        expected.append([matchup,datetime.strptime(date,'%b %d, %Y').strftime('%Y-%m-%d'),v[2],v[3],f'{float(v[6]):.1f}',v[15],v[16],v[17],v[18]])
expected.sort(key=lambda r:r[1])
assert rows == expected, 'CSV differs from official source'
assert len(rows)==479
assert len({r[1] for r in rows})==479
assert all(r[0].startswith('LAL ') for r in rows)
assert all(len(r)==9 for r in rows)
print(json.dumps({'verified':True,'rows':len(rows),'seasons':len(sources),'first_date':rows[0][1],'last_date':rows[-1][1],'source_match':'479 of 479 rows','bytes':path.stat().st_size}))
