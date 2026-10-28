"""Read-only historical snapshot audit; writes evidence only beside this script."""
import csv, json, re, hashlib, subprocess, sys, unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'outputs/nba_2025_26'
OUT = Path(__file__).resolve().parent
protected = list(SRC.iterdir()) + [ROOT / f for f in ('PRD.md','TODO.md','PROGRESS.md')]
def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
before = hashes()
run = subprocess.run([sys.executable, str(SRC/'verify_csv.py')], capture_output=True, text=True)
verification = dict(command=[sys.executable,str(SRC/'verify_csv.py')], exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr)
(OUT/'verify_run.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
raw = json.loads((SRC/'source_rows.json').read_text(encoding='utf-8'))
pages = json.loads((SRC/'source_pages.json').read_text(encoding='utf-8'))
with (SRC/'nba_2025_26_regular_season_player_game_logs.csv').open(encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f); rows=list(reader); headers=reader.fieldnames
names=defaultdict(set); ids=defaultdict(set); teams=defaultdict(set); teamnames=defaultdict(set)
pg=Counter(); pd=Counter(); games=defaultdict(set); tg=defaultdict(set); histories=defaultdict(list)
source_lookup=defaultdict(list); shape_bad=0; id_bad=0; fg_bad=0; points_bad=0; matchup_bad=0
selected=[0,1,2,3,5,6,9,18,19]
def norm(name):
    return ''.join(c for c in unicodedata.normalize('NFKD',name) if not unicodedata.combining(c) and c.isalnum()).casefold()
def canonical(v):
    return tuple(v[:4])+tuple(float(x) if x!='-' else '-' for x in v[4:])
for r in raw:
    v=r['values']; links=r['links']
    if len(v)!=26 or len(links)!=4: shape_bad+=1; continue
    pid,tid,gid=[links[i].split('/')[-1] for i in range(3)]
    if not (re.fullmatch(r'/stats/player/\d+',links[0]) and re.fullmatch(r'/stats/team/\d+',links[1]) and re.fullmatch(r'/game/00225\d{5}',links[2])): id_bad+=1
    date=datetime.strptime(v[3],'%m/%d/%Y').date().isoformat()
    names[v[0]].add(pid); ids[pid].add(v[0]); teams[pid].add(v[1]); teamnames[tid].add(v[1])
    pg[pid,gid]+=1; pd[pid,date]+=1; games[gid].add((tid,date)); tg[tid].add(gid); histories[pid].append((date,gid,v[1]))
    val=[v[i] for i in selected]; val[3]=date; source_lookup[canonical(val)].append((pid,gid,tid))
    fgm,fga=int(v[7]),int(v[8])
    if (v[9]=='-' and fga!=0) or (v[9]!='-' and (fga==0 or abs(float(v[9])-100*fgm/fga)>0.051)): fg_bad+=1
    if int(v[6])!=2*fgm+int(v[10])+int(v[13]): points_bad+=1
    if not re.fullmatch(re.escape(v[1])+r' (?:@|VS\.) [A-Z]{3}',v[2]): matchup_bad+=1
columns={}
for h in headers:
    vs=[r[h] for r in rows]; valid=[v for v in vs if v not in ('','-')]
    c={'blank':vs.count(''),'dash':vs.count('-'),'distinct':len(set(vs))}
    if h in ['MIN','PTS','FG%','REB','AST']:
        nums=[float(v) for v in valid]; c.update(min=min(nums),max=max(nums),non_integer=sum(not n.is_integer() for n in nums),decimal_places=dict(Counter(str(len(v.split('.')[1])) if '.' in v else '0' for v in valid)),invalid_numeric=sum(not re.fullmatch(r'\d+(?:\.\d+)?',v) for v in valid))
    if h=='GAMEDATE': c['invalid_dates']=sum(datetime.strptime(v,'%Y-%m-%d').date().isoformat()!=v for v in valid)
    columns[h]=c
mapped=[source_lookup[canonical([r[h] for h in headers])] for r in rows]
counts=[len(v) for v in histories.values()]
transfers=[]
for pid,hs in histories.items():
    if len(teams[pid])>1:
        hs.sort(); transitions=[{'date':b[0],'from':a[2],'to':b[2]} for a,b in zip(hs,hs[1:]) if a[2]!=b[2]]
        transfers.append({'PLAYER_ID':pid,'names':sorted(ids[pid]),'teams':sorted(teams[pid]),'transitions':transitions})
normalized=defaultdict(set)
for n in names: normalized[norm(n)].add(n)
summary={
 'executed_at_utc':datetime.now(timezone.utc).isoformat(),'python_version':sys.version,'verification':verification,
 'rows':len(rows),'source_rows':len(raw),'players_by_name':len(names),'players_by_id':len(ids),'games':len(games),'teams':len(tg),
 'date_from':min(r['GAMEDATE'] for r in rows),'date_to':max(r['GAMEDATE'] for r in rows),'columns':columns,
 'duplicate_full_rows':len(rows)-len(set(tuple(r[h] for h in headers) for r in rows)),
 'duplicate_player_game':sum(v-1 for v in pg.values()),'duplicate_player_date':sum(v-1 for v in pd.values()),
 'source_shape_bad':shape_bad,'id_link_bad':id_bad,'mapping_candidate_counts':dict(Counter(len(m) for m in mapped)),
 'source_csv_multiset_equal':Counter(canonical([r[h] for h in headers]) for r in rows)==Counter({k:len(v) for k,v in source_lookup.items()}),
 'same_name_multiple_ids':{k:sorted(v) for k,v in names.items() if len(v)>1},'same_id_multiple_names':{k:sorted(v) for k,v in ids.items() if len(v)>1},
 'normalized_name_collisions':{k:sorted(v) for k,v in normalized.items() if len(v)>1},'team_id_multiple_names':{k:sorted(v) for k,v in teamnames.items() if len(v)>1},
 'team_games':{k:len(v) for k,v in tg.items()},'games_not_two_teams_one_date':sum(len(v)!=2 or len({d for t,d in v})!=1 for v in games.values()),
 'zero_points':sum(float(r['PTS'])==0 for r in rows),'zero_minutes':sum(float(r['MIN'])==0 for r in rows),'zero_minutes_positive_points':sum(float(r['MIN'])==0 and float(r['PTS'])>0 for r in rows),
 'zero_points_positive_minutes':sum(float(r['PTS'])==0 and float(r['MIN'])>0 for r in rows),'fg_formula_bad':fg_bad,'points_formula_bad':points_bad,'matchup_format_bad':matchup_bad,
 'appearance_count_min':min(counts),'appearance_count_max':max(counts),'appearance_count_distribution':dict(sorted(Counter(counts).items())),
 'history_availability':{str(w):{'rows_with_at_least_w_prior_appearances':sum(max(n-w,0) for n in counts),'rows_without_w_prior':sum(min(n,w) for n in counts),'players_with_at_least_w_total':sum(n>=w for n in counts),'players_with_at_least_one_eligible_row':sum(n>w for n in counts)} for w in (5,10)},
 'multi_team_players':len(transfers),'transfers':transfers,
 'source_pages_metadata':{k:v for k,v in pages.items() if k!='pages'},
 'source_page_count':len(pages.get('pages',[])), 'source_page_rows_sum':sum(p['rows'] for p in pages.get('pages',[])),
 'existing_validation':json.loads((SRC/'validation.json').read_text(encoding='utf-8')),
 'protected_hashes_before':before,'protected_hashes_after':hashes(),
}
summary['protected_files_unchanged']=summary['protected_hashes_before']==summary['protected_hashes_after']
summary['zero_minute_examples']=[r for r in rows if float(r['MIN'])==0][:5]
summary['raw_minute_decimal_places']=dict(Counter(str(len(r['values'][5].split('.')[1])) if '.' in r['values'][5] else '0' for r in raw))
summary['team_abbreviation_multiple_ids']={n:sorted(t for t,ns in teamnames.items() if n in ns) for n in {n for ns in teamnames.values() for n in ns} if sum(n in ns for ns in teamnames.values())>1}
summary['pages_sequence_valid']=[p['page'] for p in pages['pages']]==list(range(534)) or [p['page'] for p in pages['pages']]==list(range(1,535))
summary['page_size_distribution']=dict(Counter(p['rows'] for p in pages['pages']))
(OUT/'audit_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ('transfers','protected_hashes_before','protected_hashes_after','source_pages_metadata','existing_validation','appearance_count_distribution')},ensure_ascii=False,indent=2))
