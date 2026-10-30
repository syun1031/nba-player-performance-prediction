"""Read-only independent arithmetic and artifact audit; never fits models."""
import json
import math
from pathlib import Path
import joblib
import numpy as np
from scripts.build_features_baseline import read_csv,sha
from scripts.same_conference_h2h import h2h_matrix
from scripts.time_validation_tuning import build_folds
from scripts.verify_two_stage import require,score
from scripts.verify_high_score_weighting import independent_alerts,close

ROOT=Path(__file__).resolve().parents[1]
KEY=lambda r:(r['player_id'],r['game_id'])


def period_rows(records,period):
    return [r for r in records if r['period']==period or (period=='CV_pooled' and r['period'] in ['F1','F2','F3'])]


def audit(out):
    meta=json.loads((out/'validation.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',out)]:
        for p,h in meta[group].items(): require(sha(base/p)==h,'Changed hash: '+p)
    prior=ROOT/'outputs/week2/same_conference_h2h_v6'
    train=read_csv(prior/'features/train.csv'); val=read_csv(prior/'features/validation.csv')
    train.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    features={KEY(r):r for r in train+val}
    targets=read_csv(out/'training_and_validation_targets.csv')
    require(len(targets)==len(features)==16683,'Target cohort')
    for r in targets:
        f=features[KEY(r)]
        require(float(r['actual_pts'])==float(f['target_pts']) and float(r['pts_mean_5'])==float(f['pts_mean_5']), 'Target source')
        require(float(r['target_delta'])==float(f['target_pts'])-float(f['pts_mean_5']),'Delta target')
    sets={name:tr for name,tr,_ in build_folds(train)}; sets['February']=train
    records=[]
    for period,n in [('CV_pooled',9023),('February',3462)]:
        rows=read_csv(out/f'predictions_{period}.csv'); require(len(rows)==n,'Scored cohort')
        controls={KEY(r):r for r in read_csv(ROOT/f'outputs/week2/high_score_weighting_v7/predictions_{period}.csv')}
        require(set(controls)=={KEY(r) for r in rows},'Different control cohort')
        for r in rows:
            f=features[KEY(r)]; c=controls[KEY(r)]
            require(r['GAMEDATE']<='2026-02-28' and r['GAMEDATE']==f['GAMEDATE'],'Scored date')
            for column,ref in [('target_pts','target_pts'),('pred_direct_w1','pred_combined_unweighted'),('pred_direct_w2','pred_combined_w2'),('pred_baseline','pred_baseline')]:
                require(float(r[column])==float(c[ref]),'Different control value')
        records+=rows
    fits=read_csv(out/'fit_log.csv'); require(len(fits)==8 and meta['fits']==8,'Fit cap')
    params=json.loads((ROOT/'outputs/week2/time_validation_tuning/common_parameters.json').read_text(encoding='utf-8'))['selected']['xgboost']['params']
    for fit in fits:
        period=fit['period']; variant=fit['variant']; tr=sets[period]
        model=joblib.load(out/'models'/f'{period}_{variant}.joblib')
        require(model.params==params and model.training_n==len(tr),'Frozen parameters/cohort')
        high=sum(float(r['target_pts'])>=30 for r in tr)
        normalizer=1+(model.multiplier-1)*high/len(tr)
        require(math.isclose(normalizer,model.weight_normalizer) and high==model.train_high_n,'Original points weights')
        require(model.trained_through<fit['validation_start'],'Train before validation')
        rows=[r for r in records if r['period']==period]
        audit_predictions(model,rows,[features[KEY(r)] for r in rows],variant)
    demo=out/'luka_demo'; old=read_csv(ROOT/'outputs/demos/luka_combined_w2_march_2026/predictions.csv')
    rows=read_csv(demo/'predictions.csv')
    require(len(rows)==17 and {KEY(r) for r in rows}=={KEY(r) for r in old},'Luka fixed cohort')
    old_index={KEY(r):r for r in old}
    f_index={KEY(r):r for r in read_csv(ROOT/'outputs/demos/luka_combined_w2_march_2026/features.csv')}
    for r in rows:
        o=old_index[KEY(r)]
        require(r['player_id']=='Luka Dončić' and r['GAMEDATE']==o['GAMEDATE'],'Luka scope')
        for col,ref in [('target_pts','actual_pts'),('actual_minutes','actual_minutes'),('pred_baseline','pred_baseline'),('pred_direct_w2','pred_weighted')]:
            require(float(r[col])==float(o[ref]),'Luka immutable source')
    for variant in ['residual_w1','residual_w2']:
        audit_predictions(joblib.load(out/'models'/f'February_{variant}.joblib'),rows,[f_index[KEY(r)] for r in rows],variant)
    check_scores(out,records); check_scores(demo,[dict(r,period='Luka_reused_demo') for r in rows])
    refs=read_csv(out/'reference_scores.csv')
    require(refs==[r for r in read_csv(ROOT/'outputs/week2/high_score_weighting_v7/reference_scores.csv') if r['variant']=='frozen9'],'Original9 reference')
    reference=read_csv(demo/'reference_scores.csv')
    check_scores_values(reference,[dict(r,period='Luka_reused_demo',target_pts=r['actual_pts'],pred_frozen9=r['pred_original']) for r in old])
    require(len(json.loads((out/'paired_intervals.json').read_text(encoding='utf-8')))==16,'Interval count')
    return dict(fits=0,saved_models_checked=8,target_rows=16683,negative_delta_n=sum(float(r['target_delta'])<0 for r in targets),
                cv_n=9023,february_n=3462,luka_reused_n=17,hashes_verified=True,metrics_verified=True,
                interval_bounds_recomputed=False,production_changed=False)


def audit_predictions(model,rows,features,variant):
    delta=model.model.predict(h2h_matrix(features,model.columns)).astype(float)
    for r,f,d in zip(rows,features,delta):
        require(float(r[variant+'_delta'])==d,'Saved delta')
        raw=float(f['pts_mean_5'])+d
        if variant+'_raw_points' in r: require(float(r[variant+'_raw_points'])==raw,'Raw points')
        require(float(r['pred_'+variant])==max(0,raw),'Added baseline/clipping')


def check_scores_values(scores,records):
    for s in scores:
        rows=period_rows(records,s['period'])
        filters={'all':lambda y:True,'under30':lambda y:y<30,'20+':lambda y:y>=20,'30+':lambda y:y>=30,
                 '0_to_under10':lambda y:0<=y<10,'10_to_under20':lambda y:10<=y<20,'20_to_under30':lambda y:20<=y<30}
        rows=[r for r in rows if filters[s['group']](float(r['target_pts']))]
        require(len(rows)==int(s['n']),'Score count')
        if not rows: continue
        y=[float(r['target_pts']) for r in rows]; p=[float(r['pred_'+s['variant']]) for r in rows]
        n,mae,rmse=score(y,p)
        for k,v in dict(mae=mae,rmse=rmse,mean_residual=sum(a-b for a,b in zip(y,p))/n,
                        overprediction_fraction=sum(b>a for a,b in zip(y,p))/n).items():
            require(close(s[k],v),'Score mismatch: '+k)


def check_scores(out,records):
    check_scores_values(read_csv(out/'scores.csv'),records)
    for s in read_csv(out/'alert_metrics.csv'):
        rows=period_rows(records,s['period'])
        for k,v in independent_alerts(rows,s['variant']).items(): require(close(s[k],v),'Alert mismatch: '+k)


if __name__=='__main__':
    print(json.dumps(audit(ROOT/'outputs/week2/residual_offset_v8'),ensure_ascii=False))
