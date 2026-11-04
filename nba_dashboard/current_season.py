"""Postgame hypothetical backtests, independent of the strict live pregame gate."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import csv
import hashlib
import json
import math
import pandas as pd
from .data import DataIssue
from .future_data import model_minutes,sha
from .future_inference import predict_features
from scripts.recent_change_features import change_features_for_request
from scripts.opponent_context import opponent_features
from scripts.team_lineup_features import TeamHistoryIndex
from scripts.merge_game_rosters import matchup_info
from scripts.prior_season_profile import PROFILE_FEATURES
from scripts.offset_prior_model import COLUMNS

FOLDER='outputs/current_season_backtest'
BASE_COLUMNS=['nba_player_id','player_id','team_id','nba_game_id','game_id','GAMEDATE',
              'actual_minutes','target_pts','pred_baseline','pred_combined_prior25_w2',
              'pred_residual25_w2','pred_route25','route25_model','lineup_source_kind','pregame_verified']


def historical_features(bundle,target,*,index=None,personal=None):
    """Never reads target PTS/MIN/REB/AST. IDs select personal history, not names."""
    day=target['GAMEDATE'];pid=str(target['nba_player_id']);team=target['team_id']
    pair,_=matchup_info(target['MATCHUP'],team);opponent=next(t for t in pair if t!=team)
    person=personal if personal is not None else [r for r in bundle['history'] if str(r['nba_player_id'])==pid]
    index=index if index is not None else TeamHistoryIndex(bundle['team_games'])
    request={k:target[k] for k in ['player_id','team_id','game_id','GAMEDATE']}
    request.update(is_home=int(target['is_home']),is_neutral_site=int(target['is_neutral_site']))
    f=change_features_for_request(person,request)
    f.update(opponent_features(index,request))
    prior=index.before(team,day);reasons=list(filter(None,f['exclusion_reason'].split(';')))
    f['team_history_count']=len(prior)
    if len(prior)<5:reasons.append('insufficient_team_history')
    if not f['opponent_eligible']:reasons.append('insufficient_opponent_history')
    if any(r['GAMEDATE']<day for r in bundle['pending']):reasons.append('unresolved_prior_games')
    starter=int(target['is_starter'])
    if starter not in (0,1):raise DataIssue('Invalid historical starter flag')
    f.update(is_starter=starter,lineup_source_kind='historical_postgame',pregame_verified=False)
    conf=bundle['conferences']
    if conf['season']!='2026-27' or not '2026-07-01'<=day<'2027-07-01':raise DataIssue('Wrong backtest season')
    same=int(conf['teams'][team]==conf['teams'][opponent]);meetings=[]
    for r in person:
        if not same or not '2026-07-01'<=r['GAMEDATE']<day or float(r['MIN'])<=0:continue
        pair,_=matchup_info(r['MATCHUP'],r['team_id'])
        if next(t for t in pair if t!=r['team_id'])==opponent:meetings.append(r)
    meetings.sort(key=lambda r:(r['GAMEDATE'],r['game_id']))
    f.update(same_conference=same,same_conf_h2h_count=len(meetings),
        same_conf_h2h_pts_last_1=float(meetings[-1]['PTS']) if meetings else None,
        same_conf_h2h_pts_mean=mean(float(r['PTS']) for r in meetings) if meetings else None,
        h2h_history_game_ids=[r['game_id'] for r in meetings])
    f.update(bundle['profiles'].get(pid,dict(zip(PROFILE_FEATURES,[None,None,None,0,0]))))
    f.update(eligible=not reasons,exclusion_reason=';'.join(dict.fromkeys(reasons)))
    return f


def backtest_rows(bundle,models):
    personal=defaultdict(list)
    for r in bundle['history']:personal[str(r['nba_player_id'])].append(r)
    index=TeamHistoryIndex(bundle['team_games']);predictions=[];excluded=[]
    for target in sorted(bundle['raw_history'],key=lambda r:(r['GAMEDATE'],r['nba_game_id'],str(r['nba_player_id']))):
        f=historical_features(bundle,target,index=index,personal=personal[str(target['nba_player_id'])])
        identity={k:target[k] for k in ['nba_player_id','player_id','team_id','nba_game_id','game_id','GAMEDATE']}
        if model_minutes(target)==0:
            excluded.append(dict(identity,reason='rounded_zero_minutes'));continue
        if not f['eligible']:
            excluded.append(dict(identity,reason=f['exclusion_reason']));continue
        p=predict_features(f,models)
        row=dict(identity,actual_minutes=float(target['MIN']),target_pts=float(target['PTS']),
            pred_baseline=p['baseline'],pred_combined_prior25_w2=p['direct_prediction'],
            pred_residual25_w2=p['offset_prediction'],pred_route25=p['prediction'],route25_model=p['model_variant'],
            lineup_source_kind='historical_postgame',pregame_verified=False,
            history_count=f['history_count'],team_history_count=f['team_history_count'],opponent_history_count=f['opponent_history_count'])
        row.update({k:f[k] for k in COLUMNS})
        predictions.append(row)
    return predictions,excluded


def write_csv(path,rows,columns):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(dict.fromkeys(columns+list(rows[0] if rows else {}))))
        writer.writeheader();writer.writerows(rows)


def load_current(root):
    root=Path(root);folder=root/FOLDER
    try:
        m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        source=json.loads((root/'outputs/future_prediction/manifest.json').read_text(encoding='utf-8'))
        if m['schema']!=1 or m['season']!='2026-27' or m['history_version']!=source['version'] or m['history_manifest_sha256']!=sha(root/'outputs/future_prediction/manifest.json'):
            raise DataIssue('本季回測與歷史資料版本不一致。')
        if m['models']!=source['models']:raise DataIssue('本季回測模型版本不一致。')
        for variant,info in m['models'].items():
            if sha(root/info['path'])!=info['sha256']:raise DataIssue('本季回測模型檔案已變動。')
        for name,digest in m['files'].items():
            if Path(name).name!=name or sha(folder/name)!=digest:raise DataIssue('本季回測資料雜湊不符。')
        frame=pd.read_csv(folder/'predictions.csv',dtype={'nba_player_id':str,'nba_game_id':str})
        if len(frame)!=m['eligible_rows']:raise DataIssue('本季回測筆數不符。')
        if not frame.empty:
            if frame.duplicated(['nba_player_id','GAMEDATE']).any() or not frame.GAMEDATE.between('2026-07-01',m['through']).all():raise DataIssue('本季回測日期或身份有誤。')
            high=frame.pred_baseline>=25
            wanted=frame.pred_residual25_w2.where(high,frame.pred_combined_prior25_w2)
            if not (frame.pred_route25-wanted).abs().le(1e-9).all():raise DataIssue('本季回測分流有誤。')
            if not frame.lineup_source_kind.eq('historical_postgame').all() or frame.pregame_verified.any():raise DataIssue('本季回測來源標示有誤。')
            numbers=frame[['target_pts','actual_minutes','pred_baseline','pred_route25','pred_combined_prior25_w2','pred_residual25_w2']].to_numpy(float)
            if not all(math.isfinite(v) and v>=0 for v in numbers.flat):raise DataIssue('本季回測包含無效數值。')
        return dict(frame=frame,manifest=m)
    except DataIssue:raise
    except (OSError,ValueError,KeyError,TypeError) as exc:raise DataIssue('本季回測尚未備齊或資料不完整。') from exc
