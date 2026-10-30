"""Fixed four-fit 2024-25 replication of the 25-feature weighted offset model."""
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.build_features_baseline import read_csv,sha
from scripts.offset_prior_2024 import OffsetPrior2024Model,COLUMNS
from scripts.historical_season_features import season_folds,FOLDS
from scripts.time_validation_tuning import FitBudget,dump
from scripts.experiment_two_stage import save_rows
from scripts.high_score_weighting import point_group_scores,alert_metrics,paired_comparison
from scripts.verify_march_2024 import audit as audit_controls

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/offset_prior25_w2_2024_25'
PRIOR=ROOT/'outputs/season_validation_2024_25'
CONTROL=ROOT/'outputs/season_validation_2024_25_three_models'
TEST=ROOT/'outputs/season_test_2024_25_three_models'
KEY=lambda r:(r['player_id'],r['game_id'])
ALL=['baseline','basic9','basic_prior14','combined_prior25_w2','residual25_w2']
PAIRS=[('combined_prior25_w2','residual25_w2'),('basic_prior14','residual25_w2')]

def assemble(model,features,controls):
    index={KEY(r):r for r in controls}
    if len(index)!=len(controls) or len(features)!=len(index) or {KEY(r) for r in features}!=set(index):raise ValueError('Changed shared cohort')
    clean=[{k:v for k,v in r.items() if not k.startswith('target_')} for r in features]
    result=model.predict_components(clean)
    fake=model.predict_components([dict(r,target_pts=99999,PTS=99999,MIN=99999) for r in clean])
    if any(not np.array_equal(result[k],fake[k]) for k in result):raise ValueError('Target leakage')
    rows=[]
    for i,f in enumerate(features):
        old=index[KEY(f)]
        if f['GAMEDATE']!=old['GAMEDATE'] or float(f['pts_mean_5'])!=float(old['pred_baseline']):raise ValueError('Baseline/date mismatch')
        if 'target_pts' in f and float(f['target_pts'])!=float(old['target_pts']):raise ValueError('Outcome mismatch')
        rows.append(dict(old,pred_residual25_w2=float(result['points'][i]),residual25_w2_delta=float(result['delta'][i]),residual25_w2_raw_points=float(result['raw_points'][i])))
    return rows

def main():
    if OUT.exists():raise ValueError('Refuse overwrite/additional fits')
    audit_controls()
    protected=[p for folder in [PRIOR,CONTROL,TEST] for p in folder.rglob('*') if p.is_file()]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    train=read_csv(PRIOR/'features/train.csv');val=read_csv(PRIOR/'features/validation.csv')
    for rows in [train,val]:rows.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    if (len(train),len(val))!=(12700,3634):raise ValueError('Changed development cohort')
    folds=season_folds(train);params=json.loads((PRIOR/'experiment_config.json').read_text(encoding='utf-8'))['params']
    oldcv=read_csv(CONTROL/'predictions_CV_pooled.csv');oldfeb=read_csv(CONTROL/'predictions_February.csv')
    oldtest=read_csv(TEST/'predictions_test.csv');tf=read_csv(TEST/'features.csv')
    if len(tf)!=7060:raise ValueError('Changed test size')
    OUT.mkdir();(OUT/'models').mkdir()
    dump(OUT/'experiment_config.json',dict(season='2024-25',columns=COLUMNS,params=params,fit_cap=4,folds=FOLDS,
        train_n=12700,cv_n=8647,february_n=3634,reused_test_n=7060,target='target_pts - pts_mean_5',
        prediction='max(0,pts_mean_5 + predicted_delta)',weight='2 if original target_pts>=30 else 1; normalize by fold mean',
        evaluation_unweighted=True,tuning_allowed=False,auto_promotion=False,independent_test=False,pairs=PAIRS))
    budget=FitBudget(4);predictions={};logs=[]
    with threadpool_limits(limits=2):
        for period,tr,va in folds+[('February',train,val)]:
            model=OffsetPrior2024Model(params).fit(tr,budget)
            controls=[r for r in oldcv if r['period']==period] if period!='February' else oldfeb
            predictions[period]=assemble(model,va,controls)
            path=OUT/'models'/f'{period}_residual25_w2.joblib';joblib.dump(model,path)
            if not np.array_equal(model.predict_components(va)['points'],joblib.load(path).predict_components(va)['points']):raise ValueError('Reload mismatch')
            logs.append(dict(period=period,fit_number=budget.count,n_train=len(tr),n_validation=len(va),n_features=25,
                train_high_n=model.train_high_n,weight_normalizer=model.weight_normalizer,train_end=model.trained_through))
        predictions['Test']=assemble(model,tf,oldtest)
    predictions['CV_pooled']=sum([predictions[n] for n,_,_ in folds],[])
    save_rows(OUT/'training_targets.csv',[dict(player_id=r['player_id'],game_id=r['game_id'],GAMEDATE=r['GAMEDATE'],target_pts=r['target_pts'],pts_mean_5=r['pts_mean_5'],target_delta=float(r['target_pts'])-float(r['pts_mean_5'])) for r in train+val])
    scores=[];alerts=[];intervals=[]
    for dataset in ['CV_pooled','February','Test']:
        rows=predictions[dataset];save_rows(OUT/f'predictions_{dataset}.csv',rows)
        for v in ALL:
            y=[float(r['target_pts']) for r in rows];p=[float(r['pred_'+v]) for r in rows]
            scores.extend(dict(dataset=dataset,variant=v,**s) for s in point_group_scores(y,p))
            alerts.append(dict(dataset=dataset,variant=v,**alert_metrics(y,p)))
        for first,second in PAIRS:
            for group in ['all','30+']:intervals.append(dict(dataset=dataset,first=first,second=second,group=group,**paired_comparison(rows,first,second,group)))
    save_rows(OUT/'scores.csv',scores);save_rows(OUT/'alert_metrics.csv',alerts);save_rows(OUT/'fit_log.csv',logs);dump(OUT/'paired_intervals.json',intervals)
    lines=['# 2024–25 增減模型＋上季表現＋高分2倍','','25欄同25–26模型，四次固定fit，既有三種直接得分模型不重訓；最後模型固定訓練至2025-01-31。',
        'CV/二月屬開發比較；三月後已使用，這次是追加研究，不是新獨立測試。沿用後續球季研究設定，屬較早球季重現；先發為賽後快照，屬日期隔離假設性回測。',
        '權重只依訓練原得分30+給2倍，各折正規化；報告誤差不加權。沒有上季資料保留缺值。沒有依本輪結果調參或更換主模型。','',
        '|期間|模型|樣本數|整體MAE|30+ MAE|','|---|---|---:|---:|---:|']
    for d in ['CV_pooled','February','Test']:
        for v in ALL:
            a=next(s for s in scores if s['dataset']==d and s['variant']==v and s['group']=='all');h=next(s for s in scores if s['dataset']==d and s['variant']==v and s['group']=='30+')
            lines.append(f"|{d}|{v}|{a['n']}|{a['mae']:.4f}|{h['mae']:.4f}|")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if budget.count!=4 or any(sha(ROOT/p)!=h for p,h in before.items()):raise ValueError('Changed controls or fit count')
    dump(OUT/'validation.json',dict(fits=4,production_changed=False,independent_test=False,source_sha256=before,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/offset_prior_2024.py',ROOT/'scripts/offset_prior_model.py']},
        output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps([s for s in scores if s['group'] in ['all','30+']],ensure_ascii=False))

if __name__=='__main__':main()
