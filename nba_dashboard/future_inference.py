"""No fitting: strict pregame feature construction and frozen two-branch inference."""
from statistics import mean
from pathlib import Path
import math
import joblib
import numpy as np
from .data import DataIssue
from .future_data import history_issues,MODEL_PATHS
from .future_source import assess_context,aware,utc_now
from scripts.opponent_context import opponent_lineup_features
from scripts.team_lineup_features import TeamHistoryIndex
from scripts.recent_game_features import season_for
from scripts.merge_game_rosters import matchup_info,ALIASES
from scripts.offset_prior_model import COLUMNS
from scripts.prior_season_profile import PROFILE_FEATURES,profile_matrix


def build_features(bundle,context,pid,now):
    fresh=assess_context(context['schedule'],*context['sources'],now)
    s=fresh['schedule'];day=s['GAMEDATE'];issues=list(fresh['reasons'])+history_issues(bundle,day,now)
    found=[(code,p) for code,t in fresh['teams'].items() for p in t['players'] if p['nba_player_id']==str(pid)]
    if len(found)!=1 or not found[0][1]['active']:
        return dict(eligible=False,exclusion_reason=';'.join(issues+['player_not_active']))
    team,person=found[0]
    id_names=bundle.get('id_names',{r['nba_player_id']:r['player_id'] for r in bundle['history']})
    def name(p):return id_names.get(p['nba_player_id'],ALIASES.get(p['name'],p['name']))
    player=name(person);opponent=s['away_team'] if team==s['home_team'] else s['home_team']
    personal_history=[r for r in bundle['history'] if r['nba_player_id']==str(pid)]
    matchup=(s['home_team']+' VS. '+s['away_team']) if team==s['home_team'] else (s['away_team']+' @ '+s['home_team'])
    target=dict(player_id=player,nba_player_id=str(pid),team_id=team,GAMEDATE=day,game_id=matchup+'|'+day,
                is_home=int(team==s['home_team'] and not fresh.get('is_neutral_site')),is_neutral_site=fresh.get('is_neutral_site'))
    starters=[name(p) for p in fresh['teams'][team]['players'] if p['nba_player_id'] in fresh['teams'][team]['starters']]
    observed=max(context['sources'],key=lambda src:aware(src['observed_at_utc']))['observed_at_utc']
    f=opponent_lineup_features(personal_history,TeamHistoryIndex(bundle['team_games']),target,starters,
        mode='pregame',confirmed=not fresh['reasons'],observed_at=observed,prediction_at=now.isoformat(),scheduled_start_at=s['scheduled_start_utc'])
    conf=bundle['conferences']
    if conf['season']!=season_for(day) or {team,opponent}-set(conf['teams']):raise DataIssue('目標球季分區尚未核對。')
    same=int(conf['teams'][team]==conf['teams'][opponent]);meetings=[]
    if same:
        for r in personal_history:
            if r['player_id']!=player or r['GAMEDATE']>=day or season_for(r['GAMEDATE'])!=season_for(day) or float(r['MIN'])<=0:continue
            pair,_=matchup_info(r['game_id'].rsplit('|',1)[0],r['team_id'])
            if next(t for t in pair if t!=r['team_id'])==opponent:meetings.append(r)
    meetings.sort(key=lambda r:(r['GAMEDATE'],r['game_id']))
    f.update(same_conference=same,same_conf_h2h_count=len(meetings),same_conf_h2h_pts_last_1=float(meetings[-1]['PTS']) if meetings else None,
        same_conf_h2h_pts_mean=mean(float(r['PTS']) for r in meetings) if meetings else None,
        h2h_history_game_ids=[r['game_id'] for r in meetings])
    f.update(bundle['profiles'].get(str(pid),dict(zip(PROFILE_FEATURES,[None,None,None,0,0]))))
    issues+=list(filter(None,f['exclusion_reason'].split(';')))
    return {**target,**f,'eligible':not issues,'exclusion_reason':';'.join(dict.fromkeys(issues))}


def route_prediction(mean5,direct,offset):
    if not all(math.isfinite(float(v)) and float(v)>=0 for v in [mean5,direct,offset]):raise ValueError('Invalid scoring output')
    variant='residual25_w2' if mean5>=25 else 'combined_prior25_w2'
    return dict(model_variant=variant,prediction=float(offset if mean5>=25 else direct),
                direct_prediction=float(direct),offset_prediction=float(offset),baseline=float(mean5))


def load_models(root,bundle):
    result={v:joblib.load(Path(root)/p) for v,p in MODEL_PATHS.items()}
    direct=result['combined_prior25_w2'];offset=result['residual25_w2']
    if direct['columns']!=COLUMNS or offset.columns!=COLUMNS or direct['trained_through']!='2026-01-31' or offset.trained_through!='2026-01-31':
        raise DataIssue('固定模型欄位或訓練日期不相符。')
    return result


def predict_features(features,models):
    x=profile_matrix([features],COLUMNS)
    direct=float(models['combined_prior25_w2']['model'].predict(x)[0])
    offset=float(models['residual25_w2'].predict_components([features])['points'][0])
    return route_prediction(float(features['pts_mean_5']),max(0,direct),offset)


def make_prediction(bundle,context,pid,models,now):
    f=build_features(bundle,context,str(pid),now)
    if not f['eligible']:raise DataIssue('目前不符合預測資格：'+f['exclusion_reason'])
    result=predict_features(f,models)
    return dict(schema=1,kind='pregame_player_points',season='2026-27',prediction_at_utc=now.isoformat(),
        nba_game_id=context['schedule']['nba_game_id'],nba_player_id=str(pid),player_id=f['player_id'],
        team_id=f['team_id'],game_id=f['game_id'],GAMEDATE=f['GAMEDATE'],scheduled_start_utc=context['schedule']['scheduled_start_utc'],
        **result,is_starter=f['is_starter'],features={k:None if f[k] is None or isinstance(f[k],float) and not math.isfinite(f[k]) else f[k] for k in COLUMNS},
        history_counts={k:f[k] for k in ['history_count','team_history_count','opponent_history_count']},
        history_version=bundle['manifest']['version'],history_collected_at_utc=bundle['manifest']['collected_at_utc'],
        models=bundle['manifest'].get('models',{}),sources=context['sources'],schedule_source=context.get('schedule_source'),
        conditional_on_appearance=True,model_trained_through='2026-01-31',current_season_forward_accuracy_available=False)


def prediction_csv_row(result):
    row={k:v for k,v in result.items() if not isinstance(v,(dict,list))}
    row.update(result['features']);row.update(result['history_counts'])
    for variant,info in result['models'].items():
        for key in ['sha256','trained_through']:row['model_'+variant+'_'+key]=info.get(key)
    receipts=dict(zip(['box','lineup'],result['sources']))
    if result.get('schedule_source'):receipts['schedule']=result['schedule_source']
    for kind,receipt in receipts.items():
        for dest,key in [('source_url','url'),('observed_at_utc','observed_at_utc'),('sha256','sha256')]:row[kind+'_'+dest]=receipt.get(key)
    return row
