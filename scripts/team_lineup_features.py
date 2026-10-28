"""Team history gate and announced-lineup fields, without inferring absences."""
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime

if __package__:
    from .recent_game_features import canonical_date, season_for, features_for_request
else:
    from recent_game_features import canonical_date, season_for, features_for_request

SPEC_VERSION='team-lineup-v1'


class TeamHistoryIndex:
    def __init__(self,team_games):
        self.games=defaultdict(list)
        seen_dates,seen_keys=set(),set()
        for game in team_games:
            day=game['GAMEDATE']
            canonical_date(day)
            team=game['team_id']
            key=team,game['game_id']
            team_day=team,day
            if key in seen_keys or team_day in seen_dates:
                raise ValueError(f'Duplicate team game/date: {key}')
            seen_keys.add(key); seen_dates.add(team_day)
            if game.get('status')=='completed':
                self.games[(team,season_for(day))].append(game)
        for records in self.games.values():
            records.sort(key=lambda r:(r['GAMEDATE'],r['game_id']))
        self.dates={key:[r['GAMEDATE'] for r in rows] for key,rows in self.games.items()}

    def before(self,team,day):
        canonical_date(day)
        key=team,season_for(day)
        stop=bisect_left(self.dates.get(key,[]),day)
        return self.games.get(key,[])[:stop]


def aware_time(value):
    if not isinstance(value,str):
        raise ValueError('Timestamp is required')
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.utcoffset() is None:
        raise ValueError('Timezone is required')
    return parsed


def lineup_for_request(team_games,target,starting_players,*,mode='pregame',confirmed=False,
                       observed_at=None,prediction_at=None,scheduled_start_at=None):
    if mode not in ('pregame','historical_postgame'):
        raise ValueError('Unknown lineup mode')
    index=team_games if isinstance(team_games,TeamHistoryIndex) else TeamHistoryIndex(team_games)
    previous=index.before(target['team_id'],target['GAMEDATE'])
    reasons=[]
    if len(previous)<5:
        reasons.append('insufficient_team_history')
    valid=(isinstance(starting_players,(list,tuple)) and len(starting_players)==5
           and all(isinstance(p,str) and p.strip() and p==p.strip() for p in starting_players))
    valid=bool(valid and len(set(starting_players))==5)
    names=sorted(starting_players) if valid else []
    if not valid:
        reasons.append('invalid_starting_lineup')
    verified=False
    if mode=='pregame':
        if confirmed is not True or not all((observed_at,prediction_at,scheduled_start_at)):
            reasons.append('missing_pregame_evidence')
        else:
            try:
                observed,prediction,start=map(aware_time,(observed_at,prediction_at,scheduled_start_at))
                if observed>prediction:
                    reasons.append('lineup_not_available_at_prediction')
                if prediction>=start:
                    reasons.append('game_already_started')
                verified=valid and observed<=prediction<start
            except (ValueError,TypeError):
                reasons.append('invalid_pregame_timestamp')
    return dict(team_history_count=len(previous),
                team_history_last_date=previous[-1]['GAMEDATE'] if previous else None,
                team_history_game_ids_5=[g['game_id'] for g in previous[-5:]],
                is_starter=int(target['player_id'] in names) if valid else None,
                confirmed_starting_players=names,
                starting_teammates=[p for p in names if p!=target['player_id']],
                lineup_source_kind=mode,pregame_verified=verified,
                lineup_eligible=not reasons,lineup_exclusion_reason=';'.join(reasons),
                live_prediction_ready=mode=='pregame' and verified and not reasons)


def features_for_lineup_request(personal_history,team_games,target,starting_players,**kwargs):
    """Full single-request gate: personal history AND team history AND lineup."""
    basic=features_for_request(personal_history,target)
    lineup=lineup_for_request(team_games,target,starting_players,**kwargs)
    good=basic['eligible'] and lineup['lineup_eligible']
    reasons=[r for r in (basic['exclusion_reason'],lineup['lineup_exclusion_reason']) if r]
    return {**basic,**lineup,'basic_eligible':basic['eligible'],'eligible':good,
            'exclusion_reason':';'.join(reasons),
            'live_prediction_ready':good and lineup['live_prediction_ready']}
