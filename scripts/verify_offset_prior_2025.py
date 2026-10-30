"""No-fit independent audit of offset components and all comparison outputs."""
import json
import math
import joblib
import numpy as np
from scripts.build_features_baseline import read_csv,sha
from scripts.compare_offset_prior_2025 import ROOT,OUT,OFFSET,PRIOR,TEST,ALL,MODELS,PAIRS,NAMES,KEY
from scripts.same_conference_h2h import h2h_matrix
from scripts.high_score_weighting import paired_comparison
from scripts.verify_residual_offset import audit as audit_offset
from scripts.verify_prior_2025_test import audit as audit_prior
from scripts.verify_two_stage import require,score
from scripts.verify_high_score_weighting import close,independent_alerts


def audit():
    audit_offset(OFFSET);audit_prior()
    meta=json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',OUT)]:
        for p,h in meta[group].items():require(sha(base/p)==h,'Changed hash '+p)
    protocol=json.loads((OUT/'comparison_protocol.json').read_text(encoding='utf-8'))
    require(protocol['models']==MODELS and protocol['new_fits_allowed']==0 and protocol['pairs']==[list(p) for p in PAIRS],'Fixed protocol')
    features={KEY(r):r for r in read_csv(TEST/'features.csv')};datasets={}
    raw={(r['PLAYER'],r['MATCHUP']+'|'+r['GAMEDATE']):r for r in read_csv(ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv')}
    for dataset,n in [('CV_pooled',9023),('February',3462),('Test',6979)]:
        rows=read_csv(OUT/f'predictions_{dataset}.csv');keys=[KEY(r) for r in rows]
        priors={KEY(r):r for r in read_csv((TEST/'predictions_test.csv') if dataset=='Test' else PRIOR/f'predictions_{dataset}.csv')}
        require(len(rows)==len(set(keys))==len(priors)==n and set(keys)==set(priors),'Common cohort')
        for r in rows:
            p=priors[KEY(r)]
            require(r['GAMEDATE']==p['GAMEDATE'] and r['period']==p['period'],'Dates/folds')
            for c in ['target_pts','pred_baseline','has_prev_season','pred_basic_prior14','pred_combined_prior25_w2']:require(float(r[c])==float(p[c]),'Unchanged prior prediction or label')
        if dataset!='Test':
            previous={KEY(r):r for r in read_csv(OFFSET/f'predictions_{dataset}.csv')}
            require(set(previous)==set(keys),'Offset common cohort')
            for r in rows:
                p=previous[KEY(r)]
                for v in MODELS[:2]:
                    for c in ['pred_'+v,v+'_delta',v+'_raw_points']:require(float(r[c])==float(p[c]),'Unchanged prior offset components')
        else:
            fs=[features[k] for k in keys]
            for v in MODELS[:2]:
                model=joblib.load(OFFSET/'models'/f'February_{v}.joblib')
                require(model.training_n==13221 and model.trained_through=='2026-01-31','Fixed offset model')
                delta=model.model.predict(h2h_matrix(fs,model.columns)).astype(float)
                for r,f,d in zip(rows,fs,delta):
                    baseline=float(f['pts_mean_5']);raw_points=baseline+d;points=max(0,raw_points)
                    require(float(r[v+'_delta'])==d and float(r[v+'_raw_points'])==raw_points and float(r['pred_'+v])==points,'Independent offset formula')
                    require(float(r['pred_baseline'])==baseline,'Offset baseline')
        human=read_csv(OUT/f'四模型逐場比較_{dataset}.csv');hi={(r['球員'],r['日期']):r for r in human}
        require(len(hi)==len(human)==n,'Human keys')
        for r in rows:
            h=hi[r['player_id'],r['GAMEDATE']];actual=float(r['target_pts']);s=raw[KEY(r)]
            require(float(h['實際得分'])==actual==float(s['PTS']) and float(h['實際上場分鐘'])==float(s['MIN']),'Human actuals')
            for v in MODELS:
                p=float(r['pred_'+v]);require(float(h[NAMES[v]])==p and float(h[NAMES[v]+'絕對誤差'])==abs(actual-p),'Human forecasts/errors')
                if v.startswith('residual'):require(float(h[NAMES[v]+'預估增減'])==float(r[v+'_delta']),'Human signed delta')
        datasets[dataset]=rows
    filters={'all':lambda y:True,'under30':lambda y:y<30,'20+':lambda y:y>=20,'30+':lambda y:y>=30,'0_to_under10':lambda y:y<10,'10_to_under20':lambda y:10<=y<20,'20_to_under30':lambda y:20<=y<30}
    for s in read_csv(OUT/'scores.csv'):
        rows=datasets[s['dataset']];g=s['group']
        selected=[r for r in rows if int(r['has_prev_season'])==int(g=='has_prior')] if g in ['has_prior','no_prior'] else [r for r in rows if filters[g](float(r['target_pts']))]
        require(len(selected)==int(s['n']),'Unweighted score n')
        if not selected:continue
        y=[float(r['target_pts']) for r in selected];p=[float(r['pred_'+s['variant']]) for r in selected];n,mae,rmse=score(y,p)
        for k,v in dict(mae=mae,rmse=rmse,mean_residual=math.fsum(a-b for a,b in zip(y,p))/n,overprediction_fraction=sum(b>a for a,b in zip(y,p))/n).items():require(close(s[k],v),'Independent score '+k)
    for s in read_csv(OUT/'alert_metrics.csv'):
        for k,v in independent_alerts(datasets[s['dataset']],s['variant']).items():require(close(s[k],v),'Independent alert/AP '+k)
    intervals=json.loads((OUT/'paired_intervals.json').read_text(encoding='utf-8'));require(len(intervals)==24,'Fixed interval count')
    for r in intervals:
        require((r['first'],r['second']) in PAIRS,'Fixed pair')
        for k,v in paired_comparison(datasets[r['dataset']],r['first'],r['second'],r['group']).items():require(r[k]==v or isinstance(v,float) and math.isclose(r[k],v,rel_tol=0,abs_tol=1e-12),'Reproduced interval '+k)
    return dict(new_fits=0,offset_models_checked=2,cv_n=9023,february_n=3462,test_n=6979,
        component_arithmetic_verified=True,old_predictions_unchanged=True,metrics_alerts_verified=True,
        human_csv_verified=True,hashes_verified=True,intervals_reproduced_with_shared_bootstrap=24,
        production_changed=False,reused_consumed_test=True)


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False))
