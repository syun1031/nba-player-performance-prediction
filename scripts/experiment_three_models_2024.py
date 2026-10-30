"""Add four basic9 fits and reuse frozen predictions for the other two models."""
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.build_features_baseline import read_csv,sha
from scripts.experiment_two_stage import save_rows,control_predictions
from scripts.three_models_2024 import BASE,base_matrix,fit_base
from scripts.historical_season_features import season_folds
from scripts.high_score_weighting import point_group_scores,alert_metrics,paired_comparison
from scripts.time_validation_tuning import FitBudget,dump
from scripts.verify_2024_season import audit as audit_previous

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'outputs/season_validation_2024_25'
OUT=ROOT/'outputs/season_validation_2024_25_three_models'
MODELS=['basic9','basic_prior14','combined_prior25_w2']
ALL=['baseline']+MODELS
NAMES=dict(baseline='五場均值',basic9='只有原九項',basic_prior14='原九項＋上季表現',combined_prior25_w2='1＋3＋同區＋上季表現＋2倍')
PAIRS=[('baseline','basic9'),('basic9','basic_prior14'),('basic9','combined_prior25_w2'),('basic_prior14','combined_prior25_w2')]


def summarize(period,rows):
    scores=[];alerts=[]
    for v in ALL:
        y=[float(r['target_pts']) for r in rows];p=[float(r['pred_'+v]) for r in rows]
        scores.extend(dict(period=period,variant=v,**s) for s in point_group_scores(y,p))
        for flag,group in [(1,'has_prior'),(0,'no_prior')]:
            selected=[r for r in rows if int(r['has_prev_season'])==flag]
            if selected:
                s=point_group_scores([float(r['target_pts']) for r in selected],[float(r['pred_'+v]) for r in selected])[0]
                scores.append(dict(period=period,variant=v,**dict(s,group=group)))
        alerts.append(dict(period=period,variant=v,**alert_metrics(y,p)))
    return scores,alerts


def main():
    if OUT.exists():raise ValueError('Refuse overwrite or additional fits')
    audit_previous()
    rawpath=ROOT/'outputs/nba_2024_25/supplement_2026_10_30/player_games_enriched.csv'
    protected=[p for p in PRIOR.rglob('*') if p.is_file()]+[rawpath,ROOT/'streamlit_app.py',ROOT/'nba_dashboard/data.py',ROOT/'nba_dashboard/views.py']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    train=read_csv(PRIOR/'features/train.csv');val=read_csv(PRIOR/'features/validation.csv')
    train.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']));val.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    if (len(train),len(val))!=(12700,3634):raise ValueError('Changed common cohort')
    folds=season_folds(train);config=json.loads((PRIOR/'experiment_config.json').read_text(encoding='utf-8'));params=config['params']
    cv=read_csv(PRIOR/'predictions_CV_pooled.csv');feb=read_csv(PRIOR/'predictions_February.csv')
    raw={(r['player_id'],r['game_id']):r for r in read_csv(rawpath)}
    OUT.mkdir();(OUT/'models').mkdir()
    dump(OUT/'experiment_config.json',dict(season='2024-25',models=MODELS,basic9_columns=BASE,params=params,fit_cap=4,
        reused_model_count=8,train_n=12700,cv_n=8647,february_n=3634,folds=config['folds'],pairs=PAIRS,
        bootstrap_iterations=2000,seed=42,multiplicity_adjusted=False,test_scoring_allowed=False,production_promoted=False,
        scope='same retrospective development replication; fixed settings originated in later-season research'))
    dump(OUT/'test_usage.json',json.loads((PRIOR/'test_usage.json').read_text(encoding='utf-8')))
    budget=FitBudget(4);predictions={};log=[];sources={}
    with threadpool_limits(limits=2):
        for period,tr,va in folds+[('February',train,val)]:
            old=[r for r in cv if r['period']==period] if period!='February' else feb
            control={v:control_predictions(va,old,'pred_'+v) for v in ALL if v!='basic9'}
            rows=[{k:r[k] for k in ['player_id','game_id','GAMEDATE','target_pts','has_prev_season']}|
                dict(period=period,**{'pred_'+v:float(p[i]) for v,p in control.items()}) for i,r in enumerate(va)]
            model=fit_base(tr,params,budget);pred=model.predict(base_matrix(va)).astype(float)
            for records in [[{k:v for k,v in r.items() if not k.startswith('target_')} for r in va],
                [dict(r,target_pts=99999,PTS=99999,MIN=99999,prev_season_pts_mean='IGNORED') for r in va]]:
                if not np.array_equal(pred,model.predict(base_matrix(records)).astype(float)):raise ValueError('Label/profile-dependent inference')
            path=OUT/'models'/f'{period}_basic9.joblib'
            bundle=dict(model=model,columns=list(BASE),params=params,n_train=len(tr),trained_through=max(r['GAMEDATE'] for r in tr),research_only=True)
            joblib.dump(bundle,path)
            if not np.array_equal(pred,joblib.load(path)['model'].predict(base_matrix(va)).astype(float)):raise ValueError('Reload mismatch')
            for r,p in zip(rows,pred):r['pred_basic9']=float(p)
            log.append(dict(period=period,variant='basic9',fit_number=budget.count,n_features=9,n_train=len(tr),n_validation=len(va),
                train_end=bundle['trained_through'],validation_start=min(r['GAMEDATE'] for r in va)))
            sources[period]={v:str((path if v=='basic9' else PRIOR/'models'/f'{period}_{v}.joblib').relative_to(ROOT)) for v in MODELS}
            predictions[period]=rows
    predictions['CV_pooled']=sum([predictions[n] for n,_,_ in folds],[])
    scores=[];alerts=[];intervals=[]
    for period,rows in predictions.items():
        s,a=summarize(period,rows);scores+=s;alerts+=a
        if period in ['CV_pooled','February']:
            save_rows(OUT/f'predictions_{period}.csv',rows)
            human=[]
            for r in rows:
                source=raw[r['player_id'],r['game_id']];actual=float(r['target_pts'])
                human.append(dict(球員=r['player_id'],日期=r['GAMEDATE'],對戰=source['MATCHUP'],實際上場分鐘=float(source['actual_minutes']),
                    實際得分=actual,五場均值=r['pred_baseline'],原九項=r['pred_basic9'],原九項加上季=r['pred_basic_prior14'],組合加上季2倍=r['pred_combined_prior25_w2'],
                    原九項絕對誤差=abs(actual-r['pred_basic9']),原九項加上季絕對誤差=abs(actual-r['pred_basic_prior14']),組合加上季2倍絕對誤差=abs(actual-r['pred_combined_prior25_w2'])))
            save_rows(OUT/f'三模型逐場預測_{period}.csv',human)
            for first,second in PAIRS:
                for group in ['all','30+']:intervals.append(dict(period=period,first=first,second=second,group=group,**paired_comparison(rows,first,second,group)))
    for name,rows in [('scores',scores),('alert_metrics',alerts),('fit_log',log)]:save_rows(OUT/f'{name}.csv',rows)
    dump(OUT/'model_sources.json',sources);dump(OUT/'paired_intervals.json',intervals);make_report(scores)
    if budget.count!=4 or any(sha(ROOT/p)!=h for p,h in before.items()):raise ValueError('Fit cap or protected source changed')
    code=[Path(__file__),ROOT/'scripts/three_models_2024.py']
    dump(OUT/'validation.json',dict(new_fits=4,reused_models=8,train_n=12700,cv_n=8647,february_n=3634,test_metrics_computed=False,
        production_promoted=False,source_sha256=before,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},
        output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps(dict(new_fits=4,scores=[s for s in scores if s['period'] in ['CV_pooled','February'] and s['group'] in ['all','30+']]),ensure_ascii=False))


def make_report(scores):
    lines=['# 2024–25三模型預測比較','',
        '依使用者加入只有原九項。三種模型共用訓練12700、CV8647及二月3634筆與固定參數；原九項新增四次fit，另兩種沿用前輪凍結模型及逐筆預測，不重訓。',
        '原九項也使用共同個人／球隊／對手五場資格樣本，以公平比較；不用其可用的額外樣本。三月後仍封存，尚未預測或評分；線上模型不變。','',
        '|期間|模型|整體MAE|RMSE|30+ MAE|<30 MAE|','|---|---|---:|---:|---:|---:|']
    for period in ['CV_pooled','February']:
        for v in ALL:
            get=lambda g:next(r for r in scores if r['period']==period and r['variant']==v and r['group']==g)
            a=get('all');values=[a['mae'],a['rmse'],get('30+')['mae'],get('under30')['mae']]
            lines.append('|'+period+'|'+NAMES[v]+'|'+'|'.join(f'{x:.4f}' for x in values)+'|')
    lines+=['','兩份繁中CSV列出三種逐場預測、實際得分與來源上場分鐘、各模型絕對誤差；全精度結果未四捨五入。',
        'scores.csv另含分箱與有無上季資料兩群；alert_metrics.csv含固定30分警示與AP（得分排序，不是機率）。',
        '16組日期區間為2000次重抽、未作多重比較校正。這是固定設定對較早球季的重現驗證，設定來自後續球季研究，不是當年鎖定模型的前瞻實驗。',
        '來源及MIN=0／姓名ID差異限制見前輪資料說明。先發為賽後快照，仍屬日期隔離的假設性回測。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
