"""Read-only audit of the four-fit prior-profile offset experiment."""
import json
import math
import joblib
import numpy as np
from scripts.build_features_baseline import read_csv,sha
from scripts.experiment_offset_prior25 import ROOT,OUT,PRIOR,CONTROL,TEST,ALL,PAIRS,KEY,NAMES
from scripts.offset_prior_model import COLUMNS
from scripts.prior_season_profile import profile_matrix
from scripts.time_validation_tuning import build_folds
from scripts.verify_offset_prior_2025 import audit as audit_controls
from scripts.verify_two_stage import require,score
from scripts.verify_high_score_weighting import close,independent_alerts
from scripts.high_score_weighting import paired_comparison


def audit():
    audit_controls()
    meta=json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',OUT)]:
        for p,h in meta[group].items():require(sha(base/p)==h,'Changed hash '+p)
    config=json.loads((OUT/'experiment_config.json').read_text(encoding='utf-8'))
    prior_config=json.loads((PRIOR/'experiment_config.json').read_text(encoding='utf-8'))
    require(config['fit_cap']==4 and meta['fits']==4 and config['columns']==COLUMNS and config['params']==prior_config['params'],'Fixed config')
    train=read_csv(PRIOR/'features/train.csv');val=read_csv(PRIOR/'features/validation.csv')
    train.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    features={KEY(r):r for r in train+val+read_csv(TEST/'features.csv')}
    targets=read_csv(OUT/'training_targets.csv');source={KEY(r):r for r in train+val}
    require(len(targets)==len(source)==16683 and {KEY(r) for r in targets}==set(source),'Training target keys')
    for r in targets:
        s=source[KEY(r)]
        require(float(r['target_pts'])==float(s['target_pts']) and float(r['pts_mean_5'])==float(s['pts_mean_5']) and float(r['target_delta'])==float(s['target_pts'])-float(s['pts_mean_5']),'Independent signed target')
    folds=build_folds(train);training={n:tr for n,tr,_ in folds};training['February']=train
    evaluation_keys={n:{KEY(r) for r in va} for n,_,va in folds};evaluation_keys['February']={KEY(r) for r in val}
    datasets={};records=[]
    for dataset,n in [('CV_pooled',9023),('February',3462),('Test',6979)]:
        rows=read_csv(OUT/f'predictions_{dataset}.csv');old={KEY(r):r for r in read_csv(CONTROL/f'predictions_{dataset}.csv')}
        keys=[KEY(r) for r in rows]
        require(len(keys)==len(set(keys))==len(old)==n and set(keys)==set(old),'Common cohort')
        for r in rows:
            o=old[KEY(r)]
            require(r['GAMEDATE']==o['GAMEDATE'] and r['period']==o['period'],'Date/fold')
            for new,previous in [('target_pts','target_pts'),('has_prev_season','has_prev_season'),('pred_baseline','pred_baseline'),('pred_residual20_w2','pred_residual_w2'),('pred_direct25_w2','pred_combined_prior25_w2')]:require(float(r[new])==float(o[previous]),'Frozen controls')
        datasets[dataset]=rows;records+=rows
    fits=read_csv(OUT/'fit_log.csv');require(len(fits)==4 and [int(r['fit_number']) for r in fits]==[1,2,3,4],'Fit sequence')
    for log in fits:
        period=log['period'];tr=training[period];m=joblib.load(OUT/'models'/f'{period}_residual25_w2.joblib')
        require(m.columns==COLUMNS and m.params==config['params'] and m.training_n==len(tr) and m.multiplier==2,'Model specification')
        require(all(m.model.get_params()[k]==v for k,v in config['params'].items()),'Actual model params')
        high=sum(float(r['target_pts'])>=30 for r in tr);norm=1+high/len(tr)
        require(m.train_high_n==high==int(log['train_high_n']) and math.isclose(m.weight_normalizer,norm) and close(log['raw_weight_normalizer'],norm),'Original points 2x weights')
        rows=[r for r in records if r['period']==period]
        require({KEY(r) for r in rows}==evaluation_keys[period] and len(rows)==int(log['n_validation']),'Fold validation set')
        require(m.trained_through==max(r['GAMEDATE'] for r in tr)<min(r['GAMEDATE'] for r in rows),'Strict model cutoff')
        check_components(m,rows,features)
    final=joblib.load(OUT/'models/February_residual25_w2.joblib');require(final.trained_through=='2026-01-31','Test frozen cutoff')
    check_components(final,datasets['Test'],features)
    filters={'all':lambda y:True,'under30':lambda y:y<30,'20+':lambda y:y>=20,'30+':lambda y:y>=30,'0_to_under10':lambda y:y<10,'10_to_under20':lambda y:10<=y<20,'20_to_under30':lambda y:20<=y<30}
    for s in read_csv(OUT/'scores.csv'):
        rows=datasets[s['dataset']];g=s['group']
        selected=[r for r in rows if int(r['has_prev_season'])==int(g=='has_prior')] if g in ['has_prior','no_prior'] else [r for r in rows if filters[g](float(r['target_pts']))]
        require(len(selected)==int(s['n']),'Metric n')
        if not selected:continue
        y=[float(r['target_pts']) for r in selected];p=[float(r['pred_'+s['variant']]) for r in selected];n,mae,rmse=score(y,p)
        for k,v in dict(mae=mae,rmse=rmse,mean_residual=math.fsum(a-b for a,b in zip(y,p))/n,overprediction_fraction=sum(b>a for a,b in zip(y,p))/n).items():require(close(s[k],v),'Unweighted metric '+k)
    for s in read_csv(OUT/'alert_metrics.csv'):
        for k,v in independent_alerts(datasets[s['dataset']],s['variant']).items():require(close(s[k],v),'Alert/AP '+k)
    for dataset,rows in datasets.items():
        human=read_csv(OUT/f'三方案逐場比較_{dataset}.csv');hi={(r['球員'],r['日期']):r for r in human};require(len(human)==len(hi)==len(rows),'Human keys')
        for r in rows:
            h=hi[r['player_id'],r['GAMEDATE']];a=float(r['target_pts'])
            require(float(h['實際得分'])==a and float(h['五場均值'])==float(r['pred_baseline']) and float(h['新增25項模型預估增減'])==float(r['residual25_w2_delta']),'Human components')
            for v in ALL[1:]:require(float(h[NAMES[v]])==float(r['pred_'+v]) and float(h[NAMES[v]+'絕對誤差'])==abs(a-float(r['pred_'+v])),'Human forecast/error')
    intervals=json.loads((OUT/'paired_intervals.json').read_text(encoding='utf-8'));require(len(intervals)==12,'Interval count')
    for r in intervals:
        require((r['first'],r['second']) in PAIRS,'Fixed pairs')
        for k,v in paired_comparison(datasets[r['dataset']],r['first'],r['second'],r['group']).items():require(r[k]==v or isinstance(v,float) and math.isclose(r[k],v,rel_tol=0,abs_tol=1e-12),'Reproduced interval '+k)
    return dict(new_fits=0,saved_models_checked=4,training_target_rows=16683,negative_delta_rows=sum(float(r['target_delta'])<0 for r in targets),
        cv_n=9023,february_n=3462,reused_test_n=6979,original_points_weights_verified=True,components_verified=True,
        frozen_controls_verified=True,metrics_alerts_verified=True,human_csv_verified=True,hashes_verified=True,
        intervals_reproduced_with_shared_bootstrap=12,production_changed=False,independent_test=False)


def check_components(m,rows,features):
    fs=[features[KEY(r)] for r in rows]
    x=profile_matrix([{k:v for k,v in f.items() if not k.startswith('target_')} for f in fs],COLUMNS)
    delta=m.model.predict(x).astype(float)
    for r,f,d in zip(rows,fs,delta):
        raw=float(f['pts_mean_5'])+d
        require(float(r['residual25_w2_delta'])==d and float(r['residual25_w2_raw_points'])==raw and float(r['pred_residual25_w2'])==max(0,raw),'Independent offset arithmetic')


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False))
