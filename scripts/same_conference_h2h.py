"""Same-conference personal head-to-head memory, strictly before target date."""
import json
from pathlib import Path
from statistics import mean

import numpy as np

from scripts.recent_game_features import canonical_date,season_for,number
from scripts.merge_game_rosters import matchup_info
from scripts.opponent_context import opponent_lineup_features,CONTEXT_FEATURES
from scripts.experiment_recent_changes import BASE
from scripts.recent_change_features import NEW_FEATURES

SNAPSHOT=Path(__file__).resolve().parents[1]/'docs/week1/conferences_2025_26.json'
CONFERENCES=json.loads(SNAPSHOT.read_text(encoding='utf-8'))
if len(CONFERENCES['teams'])!=30 or any(list(CONFERENCES['teams'].values()).count(c)!=15 for c in ['East','West']):
    raise ValueError('Incomplete conference snapshot')
H2H_FEATURES=['same_conference','same_conf_h2h_count','same_conf_h2h_pts_last_1','same_conf_h2h_pts_mean']
NULLABLE=set(H2H_FEATURES[-2:])


def identity(row):
    if 'game_id' in row:
        matchup,day=row['game_id'].rsplit('|',1)
        if day!=row['GAMEDATE']: raise ValueError('Game id/date mismatch')
    else:
        matchup=row['MATCHUP']
    team=row.get('team_id',matchup.split()[0]); pair,_=matchup_info(matchup,team)
    if not pair<=set(CONFERENCES['teams']): raise ValueError('Unknown conference team')
    return team,next(t for t in pair if t!=team)


def h2h_features(history,target):
    day=target['GAMEDATE']; canonical_date(day); season=season_for(day)
    if season!=CONFERENCES['season']: raise ValueError('Conference snapshot not verified for target season')
    team,opponent=identity(target)
    same=int(CONFERENCES['teams'][team]==CONFERENCES['teams'][opponent])
    out=dict(same_conference=same,same_conf_h2h_count=0,same_conf_h2h_pts_last_1=None,
             same_conf_h2h_pts_mean=None,h2h_opponent_id=opponent,h2h_history_game_ids=[],
             h2h_last_history_date=None,h2h_status='no_prior_meeting' if same else 'cross_conference')
    if not same: return out
    meetings=[]; seen=set()
    for r in history:
        if r['player_id']!=target['player_id'] or r['GAMEDATE']>=day or season_for(r['GAMEDATE'])!=season: continue
        _,prior_opponent=identity(r)
        if prior_opponent!=opponent: continue
        minute=number(r.get('MIN'))
        if minute is None: raise ValueError('Invalid historical head-to-head minutes')
        if minute==0: continue
        if r['GAMEDATE'] in seen: raise ValueError('Duplicate head-to-head date')
        seen.add(r['GAMEDATE'])
        points=number(r.get('PTS'),integer=True)
        if points is None: raise ValueError('Invalid head-to-head points; no fallback')
        meetings.append((r['GAMEDATE'],r['game_id'],points))
    meetings.sort()
    if meetings:
        out.update(same_conf_h2h_count=len(meetings),same_conf_h2h_pts_last_1=meetings[-1][2],
                   same_conf_h2h_pts_mean=mean(r[2] for r in meetings),h2h_history_game_ids=[r[1] for r in meetings],
                   h2h_last_history_date=meetings[-1][0],h2h_status='available')
    return out


def h2h_lineup_features(personal_history,team_games,target,starting_players,**kwargs):
    personal_history=list(personal_history)
    out=opponent_lineup_features(personal_history,team_games,target,starting_players,**kwargs)
    out.update(h2h_features(personal_history,target))
    return out


def h2h_matrix(rows,columns):
    allowed=set(BASE+NEW_FEATURES+CONTEXT_FEATURES+H2H_FEATURES)
    if not rows or len(set(columns))!=len(columns) or not set(columns)<=allowed or not set(H2H_FEATURES)<=set(columns):
        raise ValueError('Invalid head-to-head model feature whitelist')
    x=np.asarray([[np.nan if r[c] is None or r[c]=='' else float(r[c]) for c in columns] for r in rows])
    if np.isinf(x).any() or any(np.isnan(x[:,i]).any() for i,c in enumerate(columns) if c not in NULLABLE):
        raise ValueError('Unexpected missing/nonfinite feature')
    for i,r in enumerate(rows):
        flag=float(r['same_conference']); count=float(r['same_conf_h2h_count'])
        scores=[x[i,columns.index(c)] for c in H2H_FEATURES[-2:]]
        if flag not in (0,1) or count<0 or not count.is_integer() or (flag==0 and count!=0):
            raise ValueError('Invalid conference/count contract')
        if count==0:
            if not all(np.isnan(v) for v in scores): raise ValueError('No meeting must remain missing, not zero')
        elif not all(np.isfinite(v) and v>=0 for v in scores):
            raise ValueError('Known meetings require valid point features')
    return x
