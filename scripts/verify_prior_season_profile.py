"""Read-only v9 audit: independent prior sums, joins, scores and saved predictions."""
import json
import math
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from scripts.build_features_baseline import read_csv,sha
from scripts.prior_season_profile import PROFILE_FEATURES,profile_matrix
from scripts.experiment_prior_season_profile import VARIANTS,CONTROLS,PAIRS
from scripts.time_validation_tuning import build_folds
from scripts.high_score_weighting import paired_comparison
from scripts.verify_two_stage import require,score
from scripts.verify_high_score_weighting import independent_alerts,close
from scripts.verify_residual_offset import period_rows

ROOT=Path(__file__).resolve().parents[1]
KEY=lambda r:(r['player_id'],r['game_id'])


def audit(out):
    meta=json.loads((out/'validation.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',out)]:
        for p,h in meta[group].items(): require(sha(base/p)==h,'Hash changed: '+p)
    raw=read_csv(ROOT/'outputs/nba_2024_25/nba_2024_25_regular_season_player_game_logs.csv')
    require(len(raw)==26306,'Prior rows')
    grouped=defaultdict(list); seen=set()
    for r in raw:
        require('2024-10-22'<=r['GAMEDATE']<='2025-04-13','Prior date')
        key=r['PLAYER'],r['MATCHUP'],r['GAMEDATE']; require(key not in seen,'Duplicate prior key'); seen.add(key)
        if float(r['MIN'])>0: grouped[r['PLAYER']].append(r)
    expected={}
    for name,rows in grouped.items():
        n=len(rows); pts=math.fsum(float(r['PTS']) for r in rows); minutes=math.fsum(float(r['MIN']) for r in rows)
        expected[name]=[pts/n,minutes/n,pts/minutes,n,1]
    saved=read_csv(out/'player_profiles.csv')
    require(len(saved)==len(expected)==len({r['player_id'] for r in saved}),'Profile keys')
    for r in saved:
        for c,v in zip(PROFILE_FEATURES,expected[r['player_id']]): require(close(r[c],v),'Independent raw profile: '+c)
    source=ROOT/'outputs/week2/same_conference_h2h_v6/features'
    original={KEY(r):r for split in ['train','validation'] for r in read_csv(source/f'{split}.csv')}
    train=read_csv(out/'features/train.csv'); val=read_csv(out/'features/validation.csv')
    features={KEY(r):r for r in train+val}
    require((len(train),len(val),len(features))==(13221,3462,16683) and set(features)==set(original),'Feature cohort')
    for key,r in features.items():
        require({k:r[k] for k in original[key]}==original[key],'Changed original feature or label')
        values=expected.get(r['player_id'],[None,None,None,0,0])
        for c,v in zip(PROFILE_FEATURES,values): require(close(r[c],v),'Joined profile: '+c)
        require(r['GAMEDATE']>'2025-04-13' and r['GAMEDATE']<='2026-02-28','Profile before target')
    joins=read_csv(out/'player_join_audit.csv'); names={r['player_id'] for r in train+val}
    require(len(joins)==484 and {r['player_id'] for r in joins}==names,'Join audit coverage')
    for r in joins:
        values=expected.get(r['player_id'],[None,None,None,0,0])
        for c,v in zip(PROFILE_FEATURES,values): require(close(r[c],v),'Join audit values')
        require(r['match_method']==('exact' if r['player_id'] in expected else 'no_prior_record'),'Join method')
    train.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    sets={name:tr for name,tr,_ in build_folds(train)}; sets['February']=train
    evals={name:{KEY(r) for r in va} for name,_,va in build_folds(train)}; evals['February']={KEY(r) for r in val}
    records=[]
    for period,n in [('CV_pooled',9023),('February',3462)]:
        rows=read_csv(out/f'predictions_{period}.csv')
        old={KEY(r):r for r in read_csv(ROOT/f'outputs/week2/high_score_weighting_v7/predictions_{period}.csv')}
        require(len(rows)==n and len({KEY(r) for r in rows})==n and {KEY(r) for r in rows}==set(old),'Prediction cohort')
        for r in rows:
            f=features[KEY(r)]; require(r['target_pts']==f['target_pts'] and r['GAMEDATE']==f['GAMEDATE'],'Prediction labels/dates')
            require(r['has_prev_season']==f['has_prev_season'],'Prediction prior status')
            for name,column in CONTROLS.items(): require(float(r['pred_'+name])==float(old[KEY(r)][column]),'Control prediction')
        records+=rows
    config=json.loads((out/'experiment_config.json').read_text(encoding='utf-8'))
    fits=read_csv(out/'fit_log.csv')
    require(len(fits)==8 and [int(r['fit_number']) for r in fits]==list(range(1,9)) and meta['fits']==8,'Fit budget')
    params=json.loads((ROOT/'outputs/week2/time_validation_tuning/common_parameters.json').read_text(encoding='utf-8'))['selected']['xgboost']['params']
    for f in fits:
        name=f['variant']; period=f['period']; spec=VARIANTS[name]; tr=sets[period]
        model=joblib.load(out/'models'/f'{period}_{name}.joblib')
        require(model['columns']==spec['columns'] and model['params']==params==config['params'],'Model specification')
        require(all(model['model'].get_params()[k]==v for k,v in params.items()),'Actual model parameters')
        high=sum(float(r['target_pts'])>=30 for r in tr); normalizer=1+(spec['multiplier']-1)*high/len(tr)
        require(close(f['raw_weight_normalizer'],normalizer) and math.isclose(model['raw_weight_normalizer'],normalizer),'Train-only original points weights')
        require(int(f['train_high_n'])==high and int(f['n_train'])==len(tr)==model['n_train'],'Fit train cohort')
        rows=[r for r in records if r['period']==period]
        require({KEY(r) for r in rows}==evals[period] and len(rows)==int(f['n_validation']),'Fit eval cohort')
        require(model['trained_through']==max(r['GAMEDATE'] for r in tr)<min(r['GAMEDATE'] for r in rows),'Temporal fit')
        fs=[features[KEY(r)] for r in rows]
        x=profile_matrix([{k:v for k,v in r.items() if not k.startswith('target_')} for r in fs],model['columns'])
        pred=model['model'].predict(x).astype(float)
        require(np.array_equal(pred,np.asarray([float(r['pred_'+name]) for r in rows])),'Reload predictions')
    for s in read_csv(out/'scores.csv'):
        rows=period_rows(records,s['period'])
        if s['group'] in ['has_prior','no_prior']:
            rows=[r for r in rows if int(r['has_prev_season'])==int(s['group']=='has_prior')]
        else:
            filters={'all':lambda y:True,'under30':lambda y:y<30,'20+':lambda y:y>=20,'30+':lambda y:y>=30,
                     '0_to_under10':lambda y:0<=y<10,'10_to_under20':lambda y:10<=y<20,'20_to_under30':lambda y:20<=y<30}
            rows=[r for r in rows if filters[s['group']](float(r['target_pts']))]
        require(len(rows)==int(s['n']),'Score n')
        if not rows: continue
        y=[float(r['target_pts']) for r in rows]; p=[float(r['pred_'+s['variant']]) for r in rows]
        n,mae,rmse=score(y,p)
        for k,v in dict(mae=mae,rmse=rmse,mean_residual=math.fsum(a-b for a,b in zip(y,p))/n,
                        overprediction_fraction=sum(b>a for a,b in zip(y,p))/n).items(): require(close(s[k],v),'Unweighted score '+k)
    for s in read_csv(out/'alert_metrics.csv'):
        for k,v in independent_alerts(period_rows(records,s['period']),s['variant']).items(): require(close(s[k],v),'Alert/AP '+k)
    intervals=json.loads((out/'paired_intervals.json').read_text(encoding='utf-8')); require(len(intervals)==8,'Interval count')
    for r in intervals:
        require((r['first'],r['second']) in PAIRS,'Interval pair')
        again=paired_comparison(period_rows(records,r['period']),r['first'],r['second'],r['group'])
        for k,v in again.items(): require(r[k]==v or (isinstance(v,float) and math.isclose(r[k],v,rel_tol=0,abs_tol=1e-12)),'Interval reproducibility '+k)
    return dict(fits=0,saved_models_checked=8,independent_raw_profiles=len(expected),feature_rows=16683,
        matched_players=sum(name in expected for name in names),missing_players=sum(name not in expected for name in names),
        cv_n=9023,february_n=3462,metrics_and_alerts_verified=True,hashes_verified=True,
        interval_bounds_reproduced_with_shared_bootstrap=True,production_changed=False,test_scored=False)


if __name__=='__main__': print(json.dumps(audit(ROOT/'outputs/week2/prior_season_profile_v9'),ensure_ascii=False))
