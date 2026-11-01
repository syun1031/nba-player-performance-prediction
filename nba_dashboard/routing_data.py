"""Verified fixed-25 routing display; frozen four-model sources remain intact."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from .data import DataIssue,validate_predictions
from .research_data import load_research,MODELS as SINGLE_MODELS,NAMES as SINGLE_NAMES,EXPECTED,revision as old_revision

ROUTE='route25'
MODELS=list(SINGLE_MODELS)+[ROUTE]
NAMES=dict(SINGLE_NAMES,route25='25分切換方案')
SOURCE_NAMES={'combined_prior25_w2':'直接25項＋上季＋2倍','residual25_w2':'五場均值增減25項＋上季＋2倍'}
RULE='近5場平均得分 < 25：直接25項＋上季＋2倍；≥ 25：五場均值增減25項＋上季＋2倍'
HISTORY_DATES={'2025-26':('2025-10-21','2026-04-12'),'2024-25':('2024-10-22','2025-04-13')}

@dataclass
class RoutingBundle:
    predictions:dict
    histories:dict
    manifest:dict


def apply_route(frame):
    columns=['pred_baseline','pred_combined_prior25_w2','pred_residual25_w2']
    if not set(columns)<=set(frame.columns):raise DataIssue('切換所需的五場均分或模型預測缺漏。')
    a=frame[columns].to_numpy(dtype=float)
    if not np.isfinite(a).all() or (a<0).any():raise DataIssue('切換模型資料包含無效數值。')
    f=frame.copy();high=f['pred_baseline']>=25
    f['pred_'+ROUTE]=np.where(high,f.pred_residual25_w2,f.pred_combined_prior25_w2)
    f['route25_model']=np.where(high,'residual25_w2','combined_prior25_w2')
    return f


def comparison_metrics(frame,models=None,threshold=None):
    models=['baseline']+MODELS if models is None else models
    if len(set(models))!=len(models) or any(v not in ['baseline']+MODELS for v in models):raise DataIssue('無法識別比較模型。')
    data=frame if threshold is None else frame.loc[frame.target_pts>=threshold]
    columns=['model_variant','n','mae','rmse','mean_residual']
    if data.empty:return pd.DataFrame(columns=columns)
    y=data.target_pts.to_numpy(float);rows=[]
    for v in models:
        p=data['pred_'+v].to_numpy(float)
        if not np.isfinite(y).all() or not np.isfinite(p).all():raise DataIssue('比較資料含無效得分。')
        e=y-p;rows.append(dict(model_variant=v,n=len(data),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),mean_residual=float(e.mean())))
    return pd.DataFrame(rows,columns=columns)


def recent_games(history,player,day,limit=10):
    return history.loc[history.PLAYER.eq(player)&history.GAMEDATE.lt(day)&history.MIN.gt(0)].sort_values('GAMEDATE').tail(limit).reset_index(drop=True)


def revision(root):
    folder=Path(root)/'outputs/dashboard_routing'
    try:return old_revision(root)+tuple((p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(folder.iterdir()) if p.is_file())
    except OSError as exc:raise DataIssue('切換回測資料尚未備齊。') from exc


def load_routing(root):
    root=Path(root);folder=root/'outputs/dashboard_routing'
    try:
        old=load_research(root);manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest['schema']!=1 or manifest['threshold']!=25 or manifest['primary']!=ROUTE or manifest['models']!=MODELS or set(manifest['datasets'])!=set(EXPECTED):raise DataIssue('切換回測版本不相符。')
        def read(info):
            name=info['file']
            if Path(name).name!=name or not name.endswith('.csv'):raise DataIssue('無效的切換資料路徑。')
            path=folder/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:raise DataIssue('切換資料已變動，請確認版本完整。')
            return pd.read_csv(path,converters={c:float for c in ['MIN','PTS','actual_minutes','target_pts','pred_baseline']+['pred_'+v for v in MODELS]})
        frames={};histories={}
        for season,periods in EXPECTED.items():
            history=read(manifest['histories'][season]);cols=['PLAYER','TEAM','MATCHUP','GAMEDATE','MIN','PTS','actual_minutes']
            if not set(cols)<=set(history.columns) or history[cols].isna().any().any():raise DataIssue('歷史出賽資料缺漏。')
            days=history.GAMEDATE.astype(str);start,end=HISTORY_DATES[season]
            if not days.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all() or pd.to_datetime(days,format='%Y-%m-%d',errors='coerce').isna().any() or not days.between(start,end).all():raise DataIssue('歷史日期不屬於選定球季。')
            numeric=history[['MIN','PTS','actual_minutes']].to_numpy(float)
            if not np.isfinite(numeric).all() or (numeric<0).any() or history.duplicated(['PLAYER','GAMEDATE']).any():raise DataIssue('歷史資料有無效值或重複。')
            positive=history.loc[history.MIN>0].sort_values(['PLAYER','GAMEDATE']).copy()
            if not positive.actual_minutes.gt(0).all():raise DataIssue('正分鐘出賽缺少實際上場時間。')
            positive['historical_mean5']=positive.groupby('PLAYER')['PTS'].transform(lambda x:x.shift(1).rolling(5,min_periods=5).mean())
            positive['game_id']=positive.MATCHUP+'|'+positive.GAMEDATE
            context=positive.rename(columns={'PLAYER':'player_id','PTS':'source_pts','actual_minutes':'source_minutes'})[['player_id','game_id','source_pts','source_minutes','historical_mean5']]
            histories[season]=history;frames[season]={}
            if set(manifest['datasets'][season])!=set(periods):raise DataIssue('切換資料期間不完整。')
            for period,(start,end,n) in periods.items():
                info=manifest['datasets'][season][period];f=read(info)
                validate_predictions(f,expected_n=n,start=start,end=end,models=MODELS)
                if not np.isfinite(f.actual_minutes.to_numpy(float)).all() or not f.actual_minutes.gt(0).all():raise DataIssue('切換比賽分鐘無法核對。')
                ref=old.predictions[season][period]
                joined=f.merge(ref,on=['player_id','game_id'],how='outer',suffixes=('_new','_old'),validate='one_to_one',indicator=True)
                if not joined['_merge'].eq('both').all():raise DataIssue('切換方案與單模型樣本不同。')
                for c in ['GAMEDATE','target_pts','actual_minutes','period','pred_baseline']+['pred_'+v for v in SINGLE_MODELS]:
                    if not joined[c+'_new'].eq(joined[c+'_old']).all():raise DataIssue('切換方案改動了原單模型結果。')
                expected=apply_route(f)
                if not f['pred_'+ROUTE].eq(expected['pred_'+ROUTE]).all() or not f.route25_model.eq(expected.route25_model).all():raise DataIssue('切換規則與逐場結果不一致。')
                check=f.merge(context,on=['player_id','game_id'],how='left',validate='one_to_one')
                if check[['source_pts','source_minutes','historical_mean5']].isna().any().any() or not check.target_pts.eq(check.source_pts).all() or not np.allclose(check.actual_minutes,check.source_minutes,rtol=0,atol=1e-12) or not np.allclose(check.pred_baseline,check.historical_mean5,rtol=0,atol=1e-12):raise DataIssue('五場均分或實際值無法由歷史重建。')
                for group,threshold in [('all',None),('20+',20),('30+',30)]:
                    for row in comparison_metrics(f,threshold=threshold).to_dict('records'):
                        saved=info['metrics'][group][row['model_variant']]
                        if saved['n']!=row['n'] or not np.allclose([saved[k] for k in ['mae','rmse','mean_residual']],[row[k] for k in ['mae','rmse','mean_residual']],rtol=1e-12,atol=1e-12):raise DataIssue('切換比較指標與逐場結果不一致。')
                frames[season][period]=f.sort_values(['GAMEDATE','player_id','game_id']).reset_index(drop=True)
        return RoutingBundle(frames,histories,manifest)
    except DataIssue:raise
    except (OSError,ValueError,KeyError,TypeError,IndexError,pd.errors.MergeError) as exc:raise DataIssue('無法載入切換回測資料，請確認部署檔案完整。') from exc
