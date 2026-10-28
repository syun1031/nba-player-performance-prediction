"""Recent-game features shared by historical batches and single future requests."""
import math
from collections import defaultdict
from datetime import date
from statistics import mean

FEATURE_COLUMNS = ['pts_mean_5','pts_mean_10','min_mean_5','reb_mean_5',
                   'ast_mean_5','is_home','is_neutral_site','rest_days']
SPEC_VERSION = 'basic-features-v1'


def number(value, integer=False):
    try:
        n=float(value)
    except (ValueError,TypeError):
        return None
    if not math.isfinite(n) or n<0 or (integer and not n.is_integer()):
        return None
    return n


def canonical_date(day):
    d=date.fromisoformat(day)
    if d.isoformat()!=day:
        raise ValueError(f'Date must use YYYY-MM-DD: {day}')
    return d


def season_for(day):
    d=canonical_date(day)
    start=d.year if d.month>=7 else d.year-1
    return f'{start}-{str(start+1)[-2:]}'


def split_for_date(day):
    d=canonical_date(day)
    if not date(2025,10,21)<=d<=date(2026,4,12):
        raise ValueError(f'Outside approved historical range: {day}')
    return 'train' if day<='2026-01-31' else ('validation' if day<='2026-02-28' else 'test')


def features_for_request(history, target):
    """Do not read target PTS/MIN/REB/AST; same-day history is never allowed."""
    day=target['GAMEDATE']
    target_date=canonical_date(day)
    season=season_for(day)
    prior=[r for r in history if r['player_id']==target['player_id']
           and season_for(r['GAMEDATE'])==season and r['GAMEDATE']<day
           and number(r.get('MIN'))!=0]
    prior.sort(key=lambda r:(r['GAMEDATE'],r['game_id']))
    keys=[(r['GAMEDATE'],r['game_id']) for r in prior]
    if len(set(keys))!=len(keys) or len({r['GAMEDATE'] for r in prior})!=len(prior):
        raise ValueError('Ambiguous duplicate personal history date or game')
    w5,w10=prior[-5:],prior[-10:]
    last=prior[-1]['GAMEDATE'] if prior else None
    f=dict(history_count=len(prior),n_10=len(w10),last_history_date=last,
           history_game_ids_5=[r['game_id'] for r in w5],
           history_game_ids_10=[r['game_id'] for r in w10],
           **{name:None for name in FEATURE_COLUMNS})
    reasons=[]
    for flag in ('is_home','is_neutral_site'):
        v=number(target.get(flag),integer=True)
        if v not in (0,1):
            reasons.append('invalid_target_'+flag)
        else:
            f[flag]=int(v)
    if f['is_neutral_site']==1 and f['is_home']!=0:
        reasons.append('invalid_target_venue_combination')
    if last:
        f['rest_days']=(target_date-date.fromisoformat(last)).days-1
    if len(prior)<5:
        reasons.append('insufficient_history')
    else:
        for name,source,window in [
            ('pts_mean_5','PTS',w5),('pts_mean_10','PTS',w10),
            ('min_mean_5','MIN',w5),('reb_mean_5','REB',w5),('ast_mean_5','AST',w5)]:
            values=[number(r.get(source),integer=source!='MIN') for r in window]
            if any(v is None for v in values):
                reasons.append('invalid_history_'+source.lower())
            else:
                f[name]=mean(values)
    reasons=list(dict.fromkeys(reasons))
    f.update(eligible=not reasons,exclusion_reason=';'.join(reasons))
    return f


def build_feature_rows(records):
    """Produce deterministic historical audit rows, excluding zero-minute records."""
    records=sorted(records,key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    history=defaultdict(list)
    seen_keys,seen_dates=set(),set()
    output=[]
    for row in records:
        day=row['GAMEDATE']
        split=split_for_date(day)
        key=row['player_id'],row['game_id']
        player_date=row['player_id'],day
        if key in seen_keys or player_date in seen_dates:
            raise ValueError(f'Duplicate historical sample: {key}')
        seen_keys.add(key); seen_dates.add(player_date)
        if number(row.get('MIN'))==0:
            continue
        f=features_for_request(history[row['player_id']],row)
        target=number(row.get('PTS'),integer=True)
        extra=[]
        if target is None:
            extra.append('invalid_target_label')
        if number(row.get('MIN')) is None:
            extra.append('invalid_target_minutes')
        reasons=extra+([f['exclusion_reason']] if f['exclusion_reason'] else [])
        f.update(eligible=not reasons,exclusion_reason=';'.join(reasons))
        output.append(dict(player_id=row['player_id'],team_id=row['team_id'],
                           game_id=row['game_id'],GAMEDATE=day,season=season_for(day),
                           split=split,target_pts=target,spec_version=SPEC_VERSION,**f))
        # Retain invalid values rather than silently skipping them to reach older games.
        history[row['player_id']].append(row)
    return output
