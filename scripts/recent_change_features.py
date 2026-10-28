"""Five prior-game change/volatility features, leaving the v1 contract intact."""
from statistics import mean,pstdev

if __package__:
    from .recent_game_features import features_for_request,number
    from .team_lineup_features import features_for_lineup_request
else:
    from recent_game_features import features_for_request,number
    from team_lineup_features import features_for_lineup_request

NEW_FEATURES=['pts_last_1','min_mean_3','min_change_last_vs_5','pts_std_5','min_std_5']
SPEC_VERSION='recent-change-v2'


def change_features_for_request(history,target):
    history=list(history)
    basic=features_for_request(history,target)
    result=dict(basic,**dict.fromkeys(NEW_FEATURES),history_game_ids_3=basic['history_game_ids_5'][-3:],
                spec_version=SPEC_VERSION)
    if not basic['eligible']:
        return result
    wanted=set(basic['history_game_ids_5'])
    by_game={r['game_id']:r for r in history if r['player_id']==target['player_id']
             and r['GAMEDATE']<target['GAMEDATE'] and r['game_id'] in wanted and number(r.get('MIN'))!=0}
    window=[by_game[key] for key in basic['history_game_ids_5']]
    pts=[number(r['PTS'],integer=True) for r in window]
    minutes=[number(r['MIN']) for r in window]
    result.update(pts_last_1=pts[-1],min_mean_3=mean(minutes[-3:]),
                  min_change_last_vs_5=minutes[-1]-basic['min_mean_5'],
                  pts_std_5=pstdev(pts),min_std_5=pstdev(minutes))
    return result


def change_lineup_features_for_request(personal_history,team_games,target,starting_players,**kwargs):
    """Shared full request gate; adding numbers never bypasses lineup/team policy."""
    personal_history=list(personal_history)
    combined=features_for_lineup_request(personal_history,team_games,target,starting_players,**kwargs)
    change=change_features_for_request(personal_history,target)
    combined.update({key:change[key] for key in NEW_FEATURES+['history_game_ids_3']})
    combined['spec_version']=SPEC_VERSION
    combined['eligible']=combined['eligible'] and change['eligible']
    combined['live_prediction_ready']=combined['live_prediction_ready'] and change['eligible']
    return combined
