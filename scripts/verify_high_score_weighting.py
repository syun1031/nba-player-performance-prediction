"""Read-only audit: train-only weights, unweighted scores, alerts/AP and reloads."""
import json
import math
from itertools import groupby
from pathlib import Path

import joblib
import numpy as np

from scripts.build_features_baseline import read_csv,sha
from scripts.verify_two_stage import require,score
from scripts.verify_same_conference_h2h import audit as audit_h2h

ROOT=Path(__file__).resolve().parents[1]


def close(value,expected):
    return value=='' if expected is None else math.isclose(float(value),expected,rel_tol=0,abs_tol=1e-12)


def independent_alerts(rows,variant):
    pairs=[(float(r['pred_'+variant]),float(r['target_pts'])>=30) for r in rows]
    tp=sum(p>=30 and y for p,y in pairs); fp=sum(p>=30 and not y for p,y in pairs)
    fn=sum(p<30 and y for p,y in pairs); tn=sum(p<30 and not y for p,y in pairs)
    positives=tp+fn; cum_n=0; cum_high=0; ap=0.
    # Tied scores are a single threshold block, independent of row ordering.
    for _,entries in groupby(sorted(pairs,key=lambda x:x[0],reverse=True),key=lambda x:x[0]):
        block=list(entries); added=sum(y for _,y in block); cum_n+=len(block); cum_high+=added
        if positives: ap+=(cum_high/cum_n)*added/positives
    return dict(n=len(rows),actual_high_n=positives,predicted_high_n=tp+fp,tp=tp,fp=fp,fn=fn,tn=tn,
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/positives if positives else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,false_positive_rate=fp/(fp+tn) if fp+tn else None,
        prevalence=positives/len(rows),average_precision=ap if positives else None)


def audit(out):
    meta=json.loads((out/'validation.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',out)]:
        for p,h in meta[group].items(): require(sha(base/p)==h,'Changed hash: '+p)
    prior=ROOT/'outputs/week2/same_conference_h2h_v6'; audit_h2h(prior)
    train=read_csv(prior/'features/train.csv'); val=read_csv(prior/'features/validation.csv')
    features={(r['player_id'],r['game_id']):r for r in train+val}
    require(len(train)==13221 and len(val)==3462 and len(features)==16683,'Changed common features')
    cv=read_csv(out/'predictions_CV_pooled.csv'); feb=read_csv(out/'predictions_February.csv'); records=cv+feb
    require(len(cv)==9023 and len(feb)==3462 and len({(r['player_id'],r['game_id']) for r in records})==12485,'Changed scored cohort')
    for rows,file in [(cv,'predictions_CV_pooled.csv'),(feb,'predictions_February.csv')]:
        controls={(r['player_id'],r['game_id']):r for r in read_csv(prior/file)}
        require(set(controls)=={(r['player_id'],r['game_id']) for r in rows},'Different control keys')
        for r in rows:
            control=controls[(r['player_id'],r['game_id'])]
            for column,ref in [('target_pts','target_pts'),('pred_basic_unweighted','pred_common9'),
                               ('pred_combined_unweighted','pred_case13_h2h20'),('pred_baseline','pred_baseline'),('pred_frozen9','pred_frozen9')]:
                require(float(r[column])==float(control[ref]),'Changed label or control')
            require(r['GAMEDATE']<='2026-02-28','Consumed test scored')
    config=json.loads((out/'experiment_config.json').read_text(encoding='utf-8'))
    fits=read_csv(out/'fit_log.csv'); require(len(fits)==16,'Wrong fit cap')
    require([int(r['fit_number']) for r in fits]==list(range(1,17)),'Wrong fit sequence')
    for fit in fits:
        spec=config['variants'][fit['variant']]; bundle=joblib.load(out/'models'/f"{fit['period']}_{fit['variant']}.joblib")
        rows=[r for r in records if r['period']==fit['period']]
        tr=[r for r in train if r['GAMEDATE']<=fit['train_end']]
        high=sum(float(r['target_pts'])>=30 for r in tr); low=len(tr)-high; multiplier=spec['multiplier']
        raw_mean=(low+multiplier*high)/len(tr)
        require(len(tr)==bundle['n_train']==int(fit['n_train']) and high==int(fit['train_high_n']),'Wrong training label counts')
        require(bundle['trained_through']==fit['train_end']<min(r['GAMEDATE'] for r in rows),'Wrong temporal boundary')
        require(bundle['columns']==spec['columns'] and bundle['weight_multiplier']==multiplier and bundle['research_only'],'Wrong model config')
        require(close(fit['raw_weight_normalizer'],raw_mean) and math.isclose(bundle['raw_weight_normalizer'],raw_mean,abs_tol=1e-12),'Not training-only normalizer')
        require(close(fit['low_effective_weight'],1/raw_mean) and close(fit['high_effective_weight'],multiplier/raw_mean),'Wrong effective weights')
        require(close(fit['effective_weight_mean'],1) and close(fit['effective_high_share'],multiplier*high/(low+multiplier*high)),'Wrong weight summary')
        require(bundle['params']==config['params'],'Unexpected parameter change')
        for k,v in config['params'].items(): require(bundle['model'].get_params()[k]==v,'Wrong fitted parameter')
        x=np.asarray([[np.nan if features[(r['player_id'],r['game_id'])][c]=='' else float(features[(r['player_id'],r['game_id'])][c]) for c in spec['columns']] for r in rows])
        require(bundle['model'].get_booster().num_features()==len(spec['columns']),'Wrong feature count')
        require(np.array_equal(bundle['model'].predict(x).astype(float),[float(r['pred_'+fit['variant']]) for r in rows]),'Independent reload mismatch')
    scores=read_csv(out/'scores.csv')+read_csv(out/'reference_scores.csv')
    for s in scores:
        rows=cv if s['period']=='CV_pooled' else [r for r in records if r['period']==s['period']]
        predicates={'all':lambda y:True,'under30':lambda y:y<30,'20+':lambda y:y>=20,'30+':lambda y:y>=30,
                    '0_to_under10':lambda y:y<10,'10_to_under20':lambda y:10<=y<20,'20_to_under30':lambda y:20<=y<30}
        rows=[r for r in rows if predicates[s['group']](float(r['target_pts']))]
        require(len(rows)==int(s['n']),'Wrong score group count')
        if not rows:
            require(all(s[k]=='' for k in ['mae','rmse','mean_residual','overprediction_fraction']),'Fabricated empty group score')
            continue
        residual=[float(r['target_pts'])-float(r['pred_'+s['variant']]) for r in rows]
        _,mae,rmse=score([float(r['target_pts']) for r in rows],[float(r['pred_'+s['variant']]) for r in rows])
        require(close(s['mae'],mae) and close(s['rmse'],rmse),'Weighted or wrong evaluation score')
        require(close(s['mean_residual'],sum(residual)/len(rows)) and close(s['overprediction_fraction'],sum(e<0 for e in residual)/len(rows)),'Wrong residual/overprediction metric')
    for a in read_csv(out/'alert_metrics.csv'):
        rows=cv if a['period']=='CV_pooled' else [r for r in records if r['period']==a['period']]
        measured=independent_alerts(rows,a['variant'])
        require(all(close(a[k],v) for k,v in measured.items()),'Alert counts/metrics/AP mismatch')
    intervals=json.loads((out/'paired_intervals.json').read_text(encoding='utf-8'))
    require(len(intervals)==24,'Wrong paired comparison count')
    for period,rows in [('CV_pooled',cv),('February',feb)]:
        saved=[p for p in intervals if p['period']==period]
        require({(p['first'],p['second'],p['group']) for p in saved}=={(a,b,g) for a,b in config['pairs'] for g in ['all','30+']},'Missing fixed comparison')
        for p in saved:
            subset=rows if p['group']=='all' else [r for r in rows if float(r['target_pts'])>=30]
            delta=sum(abs(float(r['target_pts'])-float(r['pred_'+p['second']]))-abs(float(r['target_pts'])-float(r['pred_'+p['first']])) for r in subset)/len(subset)
            require(math.isclose(delta,p['delta_mae'],abs_tol=1e-12) and p['status']=='estimated' and p['lower_95']<=p['upper_95'],'Wrong paired point difference')
    return dict(common_feature_rows=len(features),prediction_rows=len(records),models_reloaded=16,fits=16,additional_fits=0,
        training_weights_verified=True,evaluation_unweighted_verified=True,all_scores_alerts_and_tied_ap_equal=True,
        paired_point_differences_checked=24,bootstrap_intervals_independently_recomputed=False,test_predictions_computed=False)


if __name__=='__main__':
    print(json.dumps(audit(ROOT/'outputs/week2/high_score_weighting_v7')))
