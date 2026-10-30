"""Explicitly authorized March test path; development guards stay unchanged."""
from collections import defaultdict
from statistics import mean
import json
import numpy as np
from scripts.recent_game_features import canonical_date,season_for,number
from scripts.opponent_context import opponent_lineup_features
from scripts.merge_game_rosters import matchup_info
from scripts.team_lineup_features import TeamHistoryIndex
from scripts.prior_season_profile import PROFILE_FEATURES


def march_request_features(history,team_index,target,profiles,conferences):
    day=target['GAMEDATE'];canonical_date(day)
    if conferences['season']!='2024-25' or not '2025-03-01'<=day<='2025-04-13':raise ValueError('Only authorized March-April test dates')
    pair,_=matchup_info(target['game_id'].rsplit('|',1)[0],target['team_id']);opponent=next(t for t in pair if t!=target['team_id'])
    if not pair<=set(conferences['teams']):raise ValueError('Unverified conference team')
    starters=target['starting_players'];starters=json.loads(starters) if isinstance(starters,str) else starters
    f=opponent_lineup_features(history,team_index,target,starters,mode='historical_postgame')
    same=int(conferences['teams'][target['team_id']]==conferences['teams'][opponent]);meetings=[]
    if same:
        for r in history:
            if r['player_id']!=target['player_id'] or r['GAMEDATE']>=day or season_for(r['GAMEDATE'])!='2024-25':continue
            if number(r.get('MIN'))==0:continue
            prior_pair,_=matchup_info(r['game_id'].rsplit('|',1)[0],r['team_id']);prior_opponent=next(t for t in prior_pair if t!=r['team_id'])
            if prior_opponent!=opponent:continue
            points=number(r.get('PTS'),integer=True)
            if points is None:raise ValueError('Invalid historical H2H points')
            meetings.append((r['GAMEDATE'],r['game_id'],points))
    meetings.sort()
    if len({r[0] for r in meetings})!=len(meetings):raise ValueError('Duplicate prior matchup date')
    f.update(same_conference=same,same_conf_h2h_count=len(meetings),same_conf_h2h_pts_last_1=meetings[-1][2] if meetings else None,
        same_conf_h2h_pts_mean=mean(r[2] for r in meetings) if meetings else None,h2h_history_game_ids=[r[1] for r in meetings],
        h2h_last_history_date=meetings[-1][0] if meetings else None)
    f.update(profiles.get(str(target['nba_player_id']),dict(zip(PROFILE_FEATURES,[np.nan,np.nan,np.nan,0,0]))))
    return f


def build_test_features(records,team_games,profiles,conferences):
    index=TeamHistoryIndex(team_games);history=defaultdict(list);features=[];outcomes=[];audit=[];seen=set()
    for r in sorted(records,key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id'])):
        day=r['GAMEDATE'];canonical_date(day)
        if not '2024-10-22'<=day<='2025-04-13':raise ValueError('Wrong source season')
        key=r['player_id'],day
        if key in seen:raise ValueError('Duplicate player date')
        seen.add(key);minutes=number(r['MIN'])
        if minutes is None:raise ValueError('Invalid source minutes')
        if day>='2025-03-01':
            if minutes==0:
                audit.append(dict(player_id=r['player_id'],game_id=r['game_id'],GAMEDATE=day,eligible=False,reason='zero_minutes'))
            else:
                f=march_request_features(history[r['player_id']],index,r,profiles,conferences)
                if f['is_starter']!=int(r['is_starter']):raise ValueError('Starter mismatch')
                audit.append(dict(player_id=r['player_id'],game_id=r['game_id'],GAMEDATE=day,eligible=f['eligible'],reason=f['exclusion_reason']))
                if f['eligible']:
                    labels=number(r['PTS'],integer=True)
                    if labels is None:raise ValueError('Invalid scoring outcome')
                    ids={k:r[k] for k in ['player_id','team_id','game_id','GAMEDATE','nba_player_id','nba_game_id']}
                    features.append(dict(ids,season='2024-25',split='test',**f))
                    outcomes.append(dict(ids,target_pts=labels,actual_minutes=float(r['actual_minutes']),source_MIN=minutes,MATCHUP=r['MATCHUP']))
        if minutes>0:history[r['player_id']].append(r)
    return features,outcomes,audit
