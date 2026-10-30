"""Export 2023 source data and append official 2024 metadata without changing originals."""
import json
from pathlib import Path
from collections import Counter,defaultdict
from scripts.build_features_baseline import read_csv,save_csv,sha
from scripts.fetch_historical_seasons import OUT,ROOT,page,dump

HEADERS=['PLAYER','TEAM','MATCHUP','GAMEDATE','MIN','PTS','REB','AST']


def load_games(season):
    summary=json.loads((OUT/f'collection_{season}.json').read_text(encoding='utf-8'))
    if not summary['complete']:raise ValueError('Incomplete collection '+season)
    index=json.loads((OUT/f"index_{season.replace('-','_')}.json").read_text(encoding='utf-8'))['games']
    result=[]
    for gid,day in sorted(index.items()):
        p=OUT/'cache'/season/'games'/f'{gid}.json';g=json.loads(p.read_text(encoding='utf-8'))['game']
        if g['gameId']!=gid or g['gameEt'][:10]!=day:raise ValueError('Source identity')
        result.append(g)
    counts=Counter(t['teamTricode'] for g in result for t in [g['homeTeam'],g['awayTeam']])
    if len(result)!=1230 or len(counts)!=30 or set(counts.values())!={82}:raise ValueError('Season coverage')
    return result


def save(path,rows,columns=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=columns or list(rows[0]);temp=path.with_suffix('.check.csv');save_csv(temp,rows,columns)
    if path.exists():
        if sha(path)!=sha(temp):raise ValueError('Refuse changed existing export '+str(path))
        temp.unlink()
    else:temp.replace(path)


def source_ids():
    source=ROOT/'outputs/nba_2024_25/source_rows.json';data=json.loads(source.read_text(encoding='utf-8'))
    names={};rows={}
    for r in data:
        pid=r['links'][0].rsplit('/',1)[-1];gid=r['links'][2].rsplit('/',1)[-1];v=r['values']
        if pid in names and names[pid]!=v[0]:raise ValueError('Multiple canonical names per source ID')
        names[pid]=v[0];mm,dd,year=v[3].split('/');key=v[0],v[1],f'{year}-{mm}-{dd}'
        if key in rows:raise ValueError('Duplicated original source key')
        rows[key]=(pid,gid)
    return names,rows


def rows_from_games(games,names):
    rows=[];teams=[];rosters=[];box_index={};minute_carries=[]
    for g in games:
        day=g['gameEt'][:10];home=g['homeTeam']['teamTricode'];away=g['awayTeam']['teamTricode']
        for side,other in [('homeTeam',away),('awayTeam',home)]:
            t=g[side];code=t['teamTricode'];score_other=g['awayTeam' if side=='homeTeam' else 'homeTeam']['score']
            starters=[names.get(str(p['personId']),p['name']) for p in t['players'] if p['position']]
            if len(starters)!=5:raise ValueError('Starting five')
            appeared=[p for p in t['players'] if p['seconds'] is not None and p['seconds']>0]
            roster=dict(MATCHUP=away+' @ '+home,GAMEDATE=day,TEAM=code)
            for i,p in enumerate(sorted(appeared,key=lambda p:not bool(p['position'])),1):roster[f'PLAYER_{i}']=('*' if p['position'] else '')+names.get(str(p['personId']),p['name'])
            rosters.append(roster)
            teamrow=dict(team_id=code,opponent_id=other,GAMEDATE=day,game_id=day+'|'+'|'.join(sorted([home,away])),
                nba_game_id=g['gameId'],points_for=t['score'],points_allowed=score_other,combined_points=t['score']+score_other,
                status='completed',is_neutral_site=int(g['isNeutral']),official_home_team=home)
            teams.append(teamrow)
            matchup=code+(' VS. ' if side=='homeTeam' else ' @ ')+other
            for p in t['players']:
                if p['seconds'] is None:continue
                pid=str(p['personId']);name=names.get(pid,p['name']);seconds=p['seconds']
                if str(p['minutes']).endswith(':60'):minute_carries.append(dict(nba_game_id=g['gameId'],nba_player_id=pid,raw_minutes=p['minutes'],seconds=seconds))
                r=dict(PLAYER=name,TEAM=code,MATCHUP=matchup,GAMEDATE=day,MIN=(seconds+30)//60,PTS=p['points'],REB=p['rebounds'],AST=p['assists'],
                    nba_player_id=pid,nba_game_id=g['gameId'],actual_minutes=seconds/60,official_minutes_text=p['minutes'],
                    is_starter=int(bool(p['position'])),is_home=int(side=='homeTeam' and not g['isNeutral']),is_neutral_site=int(g['isNeutral']),
                    starting_players=starters,lineup_source_kind='historical_postgame',pregame_verified=False)
                if (g['gameId'],pid) in box_index:raise ValueError('Duplicate appearance')
                box_index[g['gameId'],pid]=r;rows.append(r)
    return rows,teams,rosters,box_index,minute_carries


def main():
    names,ids=source_ids();original=ROOT/'outputs/nba_2024_25/nba_2024_25_regular_season_player_game_logs.csv'
    original_hash=sha(original);old=read_csv(original)
    boxes=load_games('2024-25');boxrows,teams,rosters,idx,carries=rows_from_games(boxes,names)
    merged=[];used=set();rounding=[];stats_differences=[]
    for r in old:
        pid,gid=ids[r['PLAYER'],r['TEAM'],r['GAMEDATE']];b=idx.get((gid,pid))
        if b is None or b['TEAM']!=r['TEAM']:raise ValueError('Original positive appearance not in official box')
        if frozenset([*r['MATCHUP'].replace(' VS. ',' @ ').split(' @ ')])!=frozenset([*b['MATCHUP'].replace(' VS. ',' @ ').split(' @ ')]):raise ValueError('Different teams')
        for col in ['PTS','REB','AST']:
            if float(r[col])!=float(b[col]):stats_differences.append(dict(nba_game_id=gid,nba_player_id=pid,column=col,original=r[col],box=b[col]))
        if float(r['MIN'])!=b['MIN']:rounding.append(dict(nba_game_id=gid,nba_player_id=pid,source_MIN=r['MIN'],box_rounded_MIN=b['MIN'],minutes_text=b['official_minutes_text']))
        merged.append(dict(r,**{k:v for k,v in b.items() if k not in HEADERS},player_id=r['PLAYER'],team_id=r['TEAM'],game_id=r['MATCHUP']+'|'+r['GAMEDATE']))
        used.add((gid,pid))
    missing=set(idx)-used
    # Legacy feeds may include active DNP players as 0:00; never add them to the original CSV.
    extra_positive=[key for key in missing if idx[key]['actual_minutes']>0]
    if extra_positive or stats_differences:raise ValueError('Appearance or outcome mismatch: '+str((extra_positive[:3],stats_differences[:3])))
    sums=defaultdict(int)
    for r in old:sums[r['GAMEDATE'],r['TEAM']]+=int(r['PTS'])
    if any(sums[t['GAMEDATE'],t['team_id']]!=t['points_for'] for t in teams):raise ValueError('Original totals differ from official score')
    supp=original.parent/'supplement_2026_10_30';supp.mkdir(exist_ok=True)
    merged.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    save(supp/'player_games_enriched.csv',merged);save(supp/'team_games.csv',sorted(teams,key=lambda r:(r['GAMEDATE'],r['team_id'])))
    rostercolumns=['MATCHUP','GAMEDATE','TEAM']+[f'PLAYER_{i}' for i in range(1,max(len(r)-3 for r in rosters)+1)]
    for r in rosters:
        for c in rostercolumns:r.setdefault(c,'')
    save(supp/'nba_2024_25_regular_season_game_rosters.csv',sorted(rosters,key=lambda r:(r['GAMEDATE'],r['TEAM'])),rostercolumns)
    previous=load_games('2023-24');prior,prior_teams,_,_,prior_carries=rows_from_games(previous,names)
    target=ROOT/'outputs/nba_2023_24';target.mkdir(exist_ok=True)
    prior.sort(key=lambda r:(r['PLAYER'],r['GAMEDATE']))
    save(target/'nba_2023_24_regular_season_player_game_logs.csv',prior,HEADERS)
    save(target/'player_games_with_official_ids.csv',prior)
    save(target/'team_games.csv',sorted(prior_teams,key=lambda r:(r['GAMEDATE'],r['team_id'])))
    props,evidence=page('https://www.nba.com/news/2024-nba-cup-group-draw-results')
    # Manually cross-checked all six groups on the dated official 2024 article.
    groups={'West A':['MIN','LAC','SAC','HOU','POR'],'West B':['OKC','PHX','LAL','UTA','SAS'],
        'West C':['DEN','DAL','NOP','GSW','MEM'],'East A':['NYK','ORL','PHI','BKN','CHA'],
        'East B':['MIL','IND','MIA','TOR','DET'],'East C':['BOS','CLE','CHI','ATL','WAS']}
    conferences={t:group.split()[0] for group,codes in groups.items() for t in codes}
    if len(conferences)!=30:raise ValueError('Incomplete 2024 conference mapping')
    dump(supp/'conferences_2024_25.json',dict(season='2024-25',teams=conferences,official_draw_groups=groups,evidence=evidence,
        verified_against='2024 official NBA Cup group draw: all 30 teams manually checked in East/West groups A/B/C',
        historical_publication_reconstruction=False))
    if sha(original)!=original_hash:raise ValueError('Original CSV was changed')
    files=[p for folder in [supp,target] for p in folder.iterdir() if p.is_file() and p.name!='validation.json']
    evidence_files=[p for p in (OUT/'cache').rglob('*.json')]
    result=dict(original_2024_csv_sha256=original_hash,original_csv_unchanged=True,season_2024_25=dict(games=1230,rows=len(merged),
        neutral_games=sum(g['isNeutral'] for g in boxes),zero_minute_rows=sum(float(r['MIN'])==0 for r in old),outcome_mismatches=stats_differences,
        minutes_rounding_differences=rounding,rendered_60_second_carries=carries),season_2023_24=dict(games=1230,rows=len(prior),players=len({r['PLAYER'] for r in prior}),
        neutral_games=sum(g['isNeutral'] for g in previous),zero_minute_rows=sum(r['MIN']==0 for r in prior),rendered_60_second_carries=prior_carries),
        test_scored=False,output_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in evidence_files},
        code_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    dump(OUT/'data_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')},ensure_ascii=False))


if __name__=='__main__':main()
