import json
import math
import joblib
import numpy as np
from scripts.experiment_offset_prior_2024 import ROOT,OUT,PRIOR,CONTROL,TEST,KEY,ALL,PAIRS
from scripts.historical_season_features import season_folds
from scripts.build_features_baseline import read_csv,sha
from scripts.prior_season_profile import profile_matrix
from scripts.offset_prior_model import COLUMNS
from scripts.verify_two_stage import require,score
from scripts.verify_high_score_weighting import close,independent_alerts
from scripts.high_score_weighting import paired_comparison

def audit():
    meta=json.loads((OUT/'validation.json').read_text(encoding='utf-8'));config=json.loads((OUT/'experiment_config.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',OUT)]:
        for p,h in meta[group].items():require(sha(base/p)==h,'Changed hash '+p)
    require(meta['fits']==config['fit_cap']==4 and config['columns']==COLUMNS,'Fixed config')
    train=read_csv(PRIOR/'features/train.csv');val=read_csv(PRIOR/'features/validation.csv')
    train.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    targets=read_csv(OUT/'training_targets.csv');sources={KEY(r):r for r in train+val}
    require(len(targets)==len(sources)==16334,'Target cohort')
    for r in targets:
        f=sources[KEY(r)];require(float(r['target_delta'])==float(f['target_pts'])-float(f['pts_mean_5']),'Signed target')
    folds=season_folds(train);training={n:tr for n,tr,_ in folds};training['February']=train
    expected={n:{KEY(r) for r in va} for n,_,va in folds};expected['February']={KEY(r) for r in val}
    features={KEY(r):r for r in train+val+read_csv(TEST/'features.csv')}
    predictions={};records=[]
    for dataset,n,source in [('CV_pooled',8647,CONTROL/'predictions_CV_pooled.csv'),('February',3634,CONTROL/'predictions_February.csv'),('Test',7060,TEST/'predictions_test.csv')]:
        rows=read_csv(OUT/f'predictions_{dataset}.csv');old={KEY(r):r for r in read_csv(source)}
        require(len(rows)==len({KEY(r) for r in rows})==len(old)==n and {KEY(r) for r in rows}==set(old),'Shared cohort')
        for r in rows:
            require(all(r[k]==v for k,v in old[KEY(r)].items()),'Unchanged controls')
        predictions[dataset]=rows;records+=rows
    logs=read_csv(OUT/'fit_log.csv');require(len(logs)==4 and [int(r['fit_number']) for r in logs]==[1,2,3,4],'Fit cap')
    for log in logs:
        period=log['period'];tr=training[period];model=joblib.load(OUT/'models'/f'{period}_residual25_w2.joblib')
        require(model.columns==COLUMNS and model.params==config['params'] and model.training_n==len(tr),'Model config')
        require(all(model.model.get_params()[k]==v for k,v in config['params'].items()),'Actual params')
        high=sum(float(r['target_pts'])>=30 for r in tr)
        require(high==model.train_high_n==int(log['train_high_n']) and math.isclose(model.weight_normalizer,1+high/len(tr)),'Original score weights')
        rows=[r for r in records if r['period']==period]
        require({KEY(r) for r in rows}==expected[period] and model.trained_through==max(r['GAMEDATE'] for r in tr)<min(r['GAMEDATE'] for r in rows),'Chronological folds')
        check(model,rows,features)
    final=joblib.load(OUT/'models/February_residual25_w2.joblib');require(final.trained_through=='2025-01-31','Frozen final cutoff');check(final,predictions['Test'],features)
    filters={'all':lambda y:True,'under30':lambda y:y<30,'20+':lambda y:y>=20,'30+':lambda y:y>=30,'0_to_under10':lambda y:y<10,'10_to_under20':lambda y:10<=y<20,'20_to_under30':lambda y:20<=y<30}
    for s in read_csv(OUT/'scores.csv'):
        rows=[r for r in predictions[s['dataset']] if filters[s['group']](float(r['target_pts']))]
        require(len(rows)==int(s['n']),'Metric count')
        if rows:
            y=[float(r['target_pts']) for r in rows];p=[float(r['pred_'+s['variant']]) for r in rows];n,mae,rmse=score(y,p)
            require(close(s['mae'],mae) and close(s['rmse'],rmse) and close(s['mean_residual'],math.fsum(a-b for a,b in zip(y,p))/n),'Independent unweighted metrics')
    for s in read_csv(OUT/'alert_metrics.csv'):
        for k,v in independent_alerts(predictions[s['dataset']],s['variant']).items():require(close(s[k],v),'Alerts '+k)
    intervals=json.loads((OUT/'paired_intervals.json').read_text(encoding='utf-8'));require(len(intervals)==12,'Intervals')
    for r in intervals:
        require((r['first'],r['second']) in PAIRS,'Fixed comparison')
        for k,v in paired_comparison(predictions[r['dataset']],r['first'],r['second'],r['group']).items():require(r[k]==v or isinstance(v,float) and math.isclose(r[k],v,abs_tol=1e-12),'Interval reproduction')
    return dict(new_fits=0,saved_models_checked=4,cv_n=8647,february_n=3634,reused_test_n=7060,hashes_verified=True,original_score_weights_verified=True,components_verified=True,controls_unchanged=True,metrics_verified=True,independent_test=False)

def check(model,rows,features):
    fs=[features[KEY(r)] for r in rows];delta=model.model.predict(profile_matrix(fs,COLUMNS)).astype(float)
    for r,f,d in zip(rows,fs,delta):
        raw=float(f['pts_mean_5'])+d
        require(float(r['residual25_w2_delta'])==d and float(r['residual25_w2_raw_points'])==raw and float(r['pred_residual25_w2'])==max(0,raw),'Independent baseline plus delta')
    clean=[{k:v for k,v in f.items() if not k.startswith('target_')} for f in fs]
    fake=[dict(f,target_pts=99999,MIN=99999,PTS=99999) for f in clean]
    require(np.array_equal(model.predict_components(clean)['points'],model.predict_components(fake)['points']),'Label-free inference')

if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False))
