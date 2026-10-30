"""Prior-date opponent scoring context; no possession/Pace claims."""
from collections import defaultdict
from statistics import mean

from scripts.recent_game_features import canonical_date,number
from scripts.team_lineup_features import TeamHistoryIndex
from scripts.recent_change_features import change_lineup_features_for_request
from scripts.merge_game_rosters import matchup_info,ALIASES

CONTEXT_FEATURES=['opp_points_allowed_mean_5','opp_combined_points_mean_5']


def build_team_games(raw_rows,rosters=None):
    games=defaultdict(dict); seen=set(); seen_team_day={}
    for r in raw_rows:
        day=canonical_date(r['GAMEDATE']).isoformat()
        # First filter the date, before touching any held-out outcome field.
        if day>'2026-02-28':
            continue
        pair,_=matchup_info(r['MATCHUP'],r['TEAM']); key=day,pair
        team=r['TEAM']; team_day=team,day
        if team_day in seen_team_day and seen_team_day[team_day]!=pair:
            raise ValueError('Ambiguous team/date opponent')
        seen_team_day[team_day]=pair
        player=ALIASES.get(r['PLAYER'],r['PLAYER']); minute=number(r['MIN']); points=number(r['PTS'],integer=True)
        if (player,day) in seen or minute is None or points is None or (minute==0 and points!=0):
            raise ValueError('Duplicate player/date or invalid outcome')
        seen.add((player,day))
        record=games[key].setdefault(team,dict(points=0,appeared=set(),zero=set()))
        name=player
        if minute>0:
            record['points']+=points; record['appeared'].add(name)
        else:
            record['zero'].add(name)
    result=[]
    for (day,pair),sides in sorted(games.items(),key=lambda v:(v[0][0],sorted(v[0][1]))):
        if set(sides)!=set(pair) or len(sides)!=2:
            raise ValueError('Missing side of physical game')
        physical_id='|'.join([day,*sorted(pair)])
        for team in sorted(pair):
            opponent=next(t for t in pair if t!=team); own=sides[team]; other=sides[opponent]
            if not own['appeared'] or not other['appeared']:
                raise ValueError('No appeared players on one side')
            if rosters is not None:
                roster=rosters.get((day,team))
                if roster is None or roster['pair']!=pair or own['appeared']-set(roster['players']) or set(roster['players'])-own['appeared']-own['zero']:
                    raise ValueError('Player logs and full roster disagree')
            result.append(dict(team_id=team,opponent_id=opponent,GAMEDATE=day,game_id=physical_id,
                               points_for=own['points'],points_allowed=other['points'],
                               combined_points=own['points']+other['points'],positive_minute_players=len(own['appeared']),
                               status='completed'))
    if rosters is not None and set(rosters)!={(r['GAMEDATE'],r['team_id']) for r in result}:
        raise ValueError('Roster team-game coverage differs')
    return result


def opponent_features(team_games,target):
    day=target['GAMEDATE']; canonical_date(day)
    if 'game_id' in target:
        matchup,id_day=target['game_id'].rsplit('|',1)
        if id_day!=day:
            raise ValueError('Target game/date mismatch')
    else:
        matchup=target['MATCHUP']
    team=target.get('team_id',matchup.split()[0]); pair,_=matchup_info(matchup,team)
    opponent=next(t for t in pair if t!=team)
    index=team_games if isinstance(team_games,TeamHistoryIndex) else TeamHistoryIndex(team_games)
    prior=index.before(opponent,day); window=prior[-5:]
    eligible=len(prior)>=5
    out=dict(opponent_id=opponent,opponent_history_count=len(prior),
             opponent_last_history_date=prior[-1]['GAMEDATE'] if prior else None,
             opponent_history_game_ids_5=[r['game_id'] for r in window],opponent_eligible=eligible,
             opponent_exclusion_reason='' if eligible else 'insufficient_opponent_history',
             **dict.fromkeys(CONTEXT_FEATURES))
    if eligible:
        for name,source in zip(CONTEXT_FEATURES,['points_allowed','combined_points']):
            values=[number(r.get(source),integer=True) for r in window]
            if any(v is None for v in values):
                raise ValueError('Invalid recent opponent outcome; no fallback')
            out[name]=mean(values)
    return out


def opponent_lineup_features(personal_history,team_games,target,starting_players,**kwargs):
    index=team_games if isinstance(team_games,TeamHistoryIndex) else TeamHistoryIndex(team_games)
    out=change_lineup_features_for_request(personal_history,index,target,starting_players,**kwargs)
    context=opponent_features(index,target); out.update(context)
    out['eligible']=out['eligible'] and context['opponent_eligible']
    out['live_prediction_ready']=out['live_prediction_ready'] and context['opponent_eligible']
    if not context['opponent_eligible']:
        out['exclusion_reason']=';'.join(filter(None,[out.get('exclusion_reason'),context['opponent_exclusion_reason']]))
    return out
