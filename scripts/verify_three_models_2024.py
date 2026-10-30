"""No-fit checks for all three saved predictors and their common predictions."""
import json
import math
import joblib
import numpy as np
from scripts.build_features_baseline import read_csv,sha
from scripts.experiment_three_models_2024 import ROOT,PRIOR,OUT,MODELS,ALL,PAIRS
from scripts.three_models_2024 import BASE,base_matrix
from scripts.prior_season_profile import profile_matrix
from scripts.historical_season_features import season_folds
from scripts.verify_2024_season import audit as previous_audit
from scripts.verify_high_score_weighting import close,independent_alerts
from scripts.verify_two_stage import require,score
from scripts.verify_residual_offset import period_rows
from scripts.high_score_weighting import paired_comparison

KEY=lambda r:(r['player_id'],r['game_id'])


def audit():
    previous_audit()
    meta=json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',OUT)]:
        for p,h in meta[group].items():require(sha(base/p)==h,'Changed hash '+p)
    train=read_csv(PRIOR/'features/train.csv');val=read_csv(PRIOR/'features/validation.csv')
    train.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    features={KEY(r):r for r in train+val};folds=season_folds(train)
    training={n:tr for n,tr,_ in folds};training['February']=train
    evals={n:{KEY(r) for r in va} for n,_,va in folds};evals['February']={KEY(r) for r in val}
    records=[]
    for period,n in [('CV_pooled',8647),('February',3634)]:
        rows=read_csv(OUT/f'predictions_{period}.csv');old={KEY(r):r for r in read_csv(PRIOR/f'predictions_{period}.csv')}
        require(len(rows)==len({KEY(r) for r in rows})==n and {KEY(r) for r in rows}==set(old),'Prediction keys')
        for r in rows:
            o=old[KEY(r)]
            for c in ['GAMEDATE','target_pts','has_prev_season','period']:require(r[c]==o[c],'Changed metadata')
            for v in ['baseline','basic_prior14','combined_prior25_w2']:require(float(r['pred_'+v])==float(o['pred_'+v]),'Changed frozen prediction')
            require(r['GAMEDATE']<='2025-02-28','Test prediction')
        records+=rows
        human=read_csv(OUT/f'三模型逐場預測_{period}.csv');require(len(human)==n,'Human CSV n')
        hi={(r['球員'],r['日期']):r for r in human};require(len(hi)==n,'Human unique keys')
        raw={(r['player_id'],r['GAMEDATE']):r for r in read_csv(ROOT/'outputs/nba_2024_25/supplement_2026_10_30/player_games_enriched.csv')}
        for r in rows:
            h=hi[r['player_id'],r['GAMEDATE']];a=float(r['target_pts']);src=raw[r['player_id'],r['GAMEDATE']]
            require(float(h['實際得分'])==a and float(h['實際上場分鐘'])==float(src['actual_minutes']) and h['對戰']==src['MATCHUP'],'Human actuals')
            for v,label,error in [('basic9','原九項','原九項絕對誤差'),('basic_prior14','原九項加上季','原九項加上季絕對誤差'),('combined_prior25_w2','組合加上季2倍','組合加上季2倍絕對誤差')]:
                p=float(r['pred_'+v]);require(float(h[label])==p and float(h[error])==abs(a-p),'Human predictions/errors')
    config=json.loads((OUT/'experiment_config.json').read_text(encoding='utf-8'));sources=json.loads((OUT/'model_sources.json').read_text(encoding='utf-8'))
    require(config['models']==MODELS and config['fit_cap']==4 and meta['new_fits']==4,'New fit cap')
    require(read_csv(OUT/'fit_log.csv') and len(read_csv(OUT/'fit_log.csv'))==4,'Fit count')
    for period,paths in sources.items():
        rows=[r for r in records if r['period']==period]
        require({KEY(r) for r in rows}==evals[period],'Fold cohort')
        fs=[features[KEY(r)] for r in rows]
        for v,path in paths.items():
            bundle=joblib.load(ROOT/path)
            require(bundle['params']==config['params'] and bundle['n_train']==len(training[period]),'Model parameters/train n')
            require(all(bundle['model'].get_params()[k]==x for k,x in config['params'].items()),'Actual parameters')
            require(bundle['trained_through']==max(r['GAMEDATE'] for r in training[period])<min(r['GAMEDATE'] for r in rows),'Temporal model cutoff')
            if v=='basic9':require(bundle['columns']==BASE,'Original nine whitelist')
            no_labels=[{k:value for k,value in r.items() if not k.startswith('target_')} for r in fs]
            x=base_matrix(no_labels) if v=='basic9' else profile_matrix(no_labels,bundle['columns'])
            pred=bundle['model'].predict(x).astype(float)
            require(np.array_equal(pred,np.asarray([float(r['pred_'+v]) for r in rows])),'Reload all three models')
    filters={'all':lambda y:True,'under30':lambda y:y<30,'20+':lambda y:y>=20,'30+':lambda y:y>=30,
        '0_to_under10':lambda y:y<10,'10_to_under20':lambda y:10<=y<20,'20_to_under30':lambda y:20<=y<30}
    for s in read_csv(OUT/'scores.csv'):
        rows=period_rows(records,s['period']);g=s['group']
        rows=[r for r in rows if int(r['has_prev_season'])==int(g=='has_prior')] if g in ['has_prior','no_prior'] else [r for r in rows if filters[g](float(r['target_pts']))]
        require(len(rows)==int(s['n']),'Metric n')
        if not rows:continue
        y=[float(r['target_pts']) for r in rows];p=[float(r['pred_'+s['variant']]) for r in rows];n,mae,rmse=score(y,p)
        for k,v in dict(mae=mae,rmse=rmse,mean_residual=math.fsum(a-b for a,b in zip(y,p))/n,overprediction_fraction=sum(b>a for a,b in zip(y,p))/n).items():require(close(s[k],v),'Unweighted metric '+k)
    for s in read_csv(OUT/'alert_metrics.csv'):
        for k,v in independent_alerts(period_rows(records,s['period']),s['variant']).items():require(close(s[k],v),'Alert/AP '+k)
    intervals=json.loads((OUT/'paired_intervals.json').read_text(encoding='utf-8'));require(len(intervals)==16,'Interval count')
    for r in intervals:
        require((r['first'],r['second']) in PAIRS,'Interval pair')
        for k,v in paired_comparison(period_rows(records,r['period']),r['first'],r['second'],r['group']).items():require(r[k]==v or isinstance(v,float) and math.isclose(r[k],v,rel_tol=0,abs_tol=1e-12),'Interval '+k)
    require((OUT/'test_usage.json').read_bytes()==(PRIOR/'test_usage.json').read_bytes(),'Test protocol changed')
    return dict(new_fits=0,saved_models_checked=12,new_saved_models=4,reused_saved_models=8,cv_n=8647,february_n=3634,
        common_predictions_verified=True,human_csv_verified=True,metrics_alerts_verified=True,hashes_verified=True,
        interval_bounds_reproduced_with_shared_bootstrap=True,test_evaluations=0,production_changed=False)


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False))
