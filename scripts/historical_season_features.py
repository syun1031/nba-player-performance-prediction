"""2024-25-only development path; original 2025-26 guards remain unchanged."""
from collections import defaultdict
from statistics import mean
import math
import json
import numpy as np
from scripts.recent_game_features import canonical_date,season_for,number
from scripts.opponent_context import opponent_lineup_features
from scripts.merge_game_rosters import matchup_info
from scripts.prior_season_profile import PROFILE_FEATURES

FOLDS=[('F1','2024-11-30','2024-12-01','2024-12-31'),('F2','2024-12-31','2025-01-01','2025-01-15'),('F3','2025-01-15','2025-01-16','2025-01-31')]


def previous_profiles(rows):
    totals=defaultdict(lambda:[0,0.,0.]);seen=set()
    for r in rows:
        day=r['GAMEDATE'];canonical_date(day)
        if not '2023-10-24'<=day<='2024-04-14':raise ValueError('Wrong previous season')
        pid=str(r['nba_player_id']);key=pid,day
        if key in seen:raise ValueError('Duplicate previous player date')
        seen.add(key);minutes=number(r['MIN']);pts=number(r['PTS'],integer=True)
        if minutes is None or pts is None:raise ValueError('Invalid prior statistics')
        if minutes==0:continue
        t=totals[pid];t[0]+=1;t[1]+=pts;t[2]+=minutes
    return {pid:dict(zip(PROFILE_FEATURES,[p/n,m/n,p/m,n,1])) for pid,(n,p,m) in sorted(totals.items())}


def development_rows(rows):
    result=[]
    for r in rows:
        day=r['GAMEDATE'];canonical_date(day)
        if not '2024-10-22'<=day<='2025-04-13':raise ValueError('Wrong target season')
        if day>'2025-02-28':continue
        result.append(r)
    return result


def request_features(history,team_index,target,profiles,conferences):
    day=target['GAMEDATE'];canonical_date(day)
    if conferences['season']!='2024-25' or not '2024-10-22'<=day<='2025-02-28':raise ValueError('Only approved development season')
    pair,_=matchup_info(target['game_id'].rsplit('|',1)[0],target['team_id']);opponent=next(t for t in pair if t!=target['team_id'])
    if not pair<=set(conferences['teams']):raise ValueError('Unverified conference team')
    starters=target['starting_players']
    if isinstance(starters,str):starters=json.loads(starters)
    f=opponent_lineup_features(history,team_index,target,starters,mode='historical_postgame')
    same=int(conferences['teams'][target['team_id']]==conferences['teams'][opponent]);meetings=[]
    if same:
        for r in history:
            if r['player_id']!=target['player_id'] or r['GAMEDATE']>=day or season_for(r['GAMEDATE'])!='2024-25':continue
            prior_pair,_=matchup_info(r['game_id'].rsplit('|',1)[0],r['team_id'])
            prior_opp=next(t for t in prior_pair if t!=r['team_id'])
            if prior_opp!=opponent or number(r.get('MIN'))==0:continue
            pts=number(r.get('PTS'),integer=True)
            if pts is None:raise ValueError('Invalid prior matchup points')
            meetings.append((r['GAMEDATE'],r['game_id'],pts))
    meetings.sort()
    if len({r[0] for r in meetings})!=len(meetings):raise ValueError('Duplicate matchup date')
    f.update(same_conference=same,same_conf_h2h_count=len(meetings),same_conf_h2h_pts_last_1=meetings[-1][2] if meetings else None,
        same_conf_h2h_pts_mean=mean(r[2] for r in meetings) if meetings else None,h2h_history_game_ids=[r[1] for r in meetings],
        h2h_last_history_date=meetings[-1][0] if meetings else None)
    f.update(profiles.get(str(target['nba_player_id']),dict(zip(PROFILE_FEATURES,[np.nan,np.nan,np.nan,0,0]))))
    return f


def season_folds(rows):
    keys=set()
    for r in rows:
        day=r['GAMEDATE'];canonical_date(day);key=r['player_id'],r['game_id']
        if not '2024-10-22'<=day<='2025-01-31' or key in keys:raise ValueError('Invalid training cohort/date')
        keys.add(key)
    folds=[]
    for name,end,start,last in FOLDS:
        tr=[r for r in rows if r['GAMEDATE']<=end];va=[r for r in rows if start<=r['GAMEDATE']<=last]
        if not tr or not va or max(r['GAMEDATE'] for r in tr)>=min(r['GAMEDATE'] for r in va):raise ValueError('Invalid fold')
        folds.append((name,tr,va))
    return folds
