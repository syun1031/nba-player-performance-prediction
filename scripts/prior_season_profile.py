"""Prior-season fixed numeric player profiles; no player-ID encoding."""
from collections import defaultdict
from datetime import date
import math
import numpy as np
from scripts.high_score_weighting import COMBINED
from scripts.same_conference_h2h import H2H_FEATURES

PROFILE_FEATURES=['prev_season_pts_mean','prev_season_min_mean','prev_season_pts_per_min',
                  'prev_season_games','has_prev_season']


def day(value):
    try:
        parsed=date.fromisoformat(value)
        if parsed.isoformat()!=value: raise ValueError('Noncanonical date')
        return value
    except (TypeError,ValueError) as e: raise ValueError('Invalid date') from e


def build_profiles(rows):
    totals=defaultdict(lambda:[0,0.,0.]); seen=set()
    for r in rows:
        d=day(r['GAMEDATE']); name=r['PLAYER']
        if not '2024-10-22'<=d<='2025-04-13' or not name or name.strip()!=name:
            raise ValueError('Wrong profile season or player name')
        key=name,r['MATCHUP'],d
        if key in seen: raise ValueError('Duplicate prior player game')
        seen.add(key)
        minutes=float(r['MIN']); points=float(r['PTS'])
        if not math.isfinite(minutes) or not math.isfinite(points) or min(minutes,points)<0 or not points.is_integer():
            raise ValueError('Invalid prior minutes or points')
        if minutes==0: continue
        t=totals[name]; t[0]+=1; t[1]+=points; t[2]+=minutes
    return {name:dict(zip(PROFILE_FEATURES,[points/n,minutes/n,points/minutes,n,1]))
            for name,(n,points,minutes) in sorted(totals.items())}


def augment(rows,profiles):
    result=[]
    for r in rows:
        if not '2025-10-21'<=day(r['GAMEDATE'])<='2026-02-28':
            raise ValueError('Only current development dates allowed')
        if any(k in r for k in PROFILE_FEATURES): raise ValueError('Do not overwrite existing profiles')
        p=profiles.get(r['player_id'],dict(zip(PROFILE_FEATURES,[np.nan,np.nan,np.nan,0,0])))
        result.append(dict(r,**p))
    return result


def profile_matrix(rows,columns):
    if not rows or len(set(columns))!=len(columns) or any(c not in COMBINED+PROFILE_FEATURES for c in columns):
        raise ValueError('Invalid feature whitelist')
    missing=set(PROFILE_FEATURES[:3]+H2H_FEATURES[2:])
    result=[]
    for r in rows:
        values=[]
        for c in columns:
            value=r[c]
            number=np.nan if value is None or value=='' else float(value)
            if math.isinf(number) or (math.isnan(number) and c not in missing): raise ValueError('Invalid feature value')
            values.append(number)
        result.append(values)
    return np.asarray(result,dtype=float)
