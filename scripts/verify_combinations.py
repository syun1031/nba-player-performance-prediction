"""Read-only four-combination artifact audit; no training or test-period scoring."""
import json
import math
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np

from scripts.build_features_baseline import read_csv,sha
from scripts.combination_models import CASES
from scripts.verify_two_stage import require,score
from scripts.verify_opponent_context import audit as audit_context

ROOT=Path(__file__).resolve().parents[1]


def audit(out):
    meta=json.loads((out/'validation.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',out)]:
        for p,h in meta[group].items(): require(sha(base/p)==h,'Changed hash: '+p)
    prior=ROOT/'outputs/week2/opponent_context_v4'
    audit_context(prior)
    raw={}
    for r in read_csv(ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv'):
        if r['GAMEDATE']>'2026-02-28': continue
        raw[(r['PLAYER'],r['MATCHUP']+'|'+r['GAMEDATE'])]=r
    features={}; n=0
    for split,size in [('train',13221),('validation',3462)]:
        old={(r['player_id'],r['game_id']):r for r in read_csv(prior/f'features/{split}.csv')}
        rows=read_csv(out/f'features/{split}.csv')
        require(len(rows)==len(old)==size,'Different common cohort size')
        require({(r['player_id'],r['game_id']) for r in rows}==set(old),'Different common cohort keys')
        for r in rows:
            key=r['player_id'],r['game_id']; ref=old[key]; source=raw[key]
            require(all(r[k]==v for k,v in ref.items()),'Changed common feature values')
            require(float(r['target_minutes'])==float(source['MIN'])>0,'Wrong minute label')
            require(float(r['target_pts'])==float(source['PTS']),'Wrong point label')
            require(math.isclose(float(r['target_ppm']),float(source['PTS'])/float(source['MIN']),abs_tol=1e-12),'Wrong rate label')
            features[key]=r; n+=1
    cv=read_csv(out/'predictions_CV_pooled.csv'); feb=read_csv(out/'predictions_February.csv'); records=cv+feb
    require(len(cv)==9023 and len(feb)==3462 and len({(r['player_id'],r['game_id']) for r in records})==12485,'Different prediction cohort')
    for rows,file in [(cv,'predictions_CV_pooled.csv'),(feb,'predictions_February.csv')]:
        old={(r['player_id'],r['game_id']):r for r in read_csv(prior/file)}
        require(set(old)=={(r['player_id'],r['game_id']) for r in rows},'Different evaluation keys')
        for r in rows:
            key=r['player_id'],r['game_id']; ref=old[key]
            require(r['GAMEDATE']<='2026-02-28','Test-period prediction found')
            for new_col,old_col in [('target_pts','target_pts'),('pred_case13','pred_opponent16'),
                                    ('pred_baseline','pred_baseline'),('pred_frozen9','pred_frozen9')]:
                require(float(r[new_col])==float(ref[old_col]),'Case13/reference mismatch')
            require(float(r['target_minutes'])==float(features[key]['target_minutes']),'Wrong scored minute label')
    log=read_csv(out/'fit_log.csv')
    require(len(log)==28 and [int(r['fit_number']) for r in log]==list(range(1,29)),'Wrong fit count/sequence')
    bundles=0; underlying=0
    for period in ['F1','F2','F3','February']:
        rows=[r for r in records if r['period']==period]
        for case,spec in CASES.items():
            entries=[r for r in log if r['period']==period and r['case']==case]
            require(len(entries)==spec['fits'],'Wrong number of case fits')
            bundle=joblib.load(out/'models'/f'{period}_{case}.joblib'); model=bundle['model']
            require(bundle['columns']==spec['columns'] and bundle['mode']==spec['mode'] and bundle['case']==case and bundle['research_only'],'Wrong saved case specification')
            train=[r for r in features.values() if r['split']=='train' and r['GAMEDATE']<=bundle['trained_through']]
            require(len(train)==bundle['n_train']==int(entries[0]['n_train']),'Different model train cohort')
            require(bundle['trained_through']<min(r['GAMEDATE'] for r in rows),'Future-trained model')
            x=np.asarray([[float(features[(r['player_id'],r['game_id'])][c]) for c in spec['columns']] for r in rows])
            if spec['mode']=='two_stage':
                require(model.columns==spec['columns'] and model.training_n==len(train),'Wrong inner model specification')
                require(math.isclose(model.weight_normalizer,sum(float(r['target_minutes']) for r in train)/len(train),abs_tol=1e-12),'Validation minutes used in normalization')
                raw_minutes=model.minutes_model.predict(x).astype(float); raw_rate=model.rate_model.predict(x).astype(float)
                minute=np.maximum(raw_minutes,0); rate=np.maximum(raw_rate,0); pred=minute*rate
                for component,value in [('minutes',minute),('rate',rate),('raw_minutes',raw_minutes),('raw_rate',raw_rate)]:
                    require(np.array_equal(value,[float(r[case+'_'+component]) for r in rows]),'Two-stage component mismatch')
                require(model.minutes_model.get_booster().num_features()==model.rate_model.get_booster().num_features()==len(spec['columns']),'Unselected features in model')
                underlying+=2
            else:
                pred=model.predict(x).astype(float); underlying+=1
                require(model.get_booster().num_features()==len(spec['columns']),'Wrong direct model feature count')
            require(np.array_equal(pred,[float(r['pred_'+case]) for r in rows]),'Saved predictions differ after independent reload')
            bundles+=1
    scores=read_csv(out/'scores.csv'); references=read_csv(out/'reference_scores.csv')
    for s in scores+references:
        rows=cv if s['period']=='CV_pooled' else [r for r in records if r['period']==s['period']]
        if s['group']!='all': rows=[r for r in rows if float(r['target_pts'])>=int(s['group'][:-1])]
        count,mae,rmse=score([float(r['target_pts']) for r in rows],[float(r['pred_'+s['variant']]) for r in rows])
        require(count==int(s['n']) and math.isclose(mae,float(s['mae']),abs_tol=1e-12) and math.isclose(rmse,float(s['rmse']),abs_tol=1e-12),'Independent points score mismatch')
    for s in read_csv(out/'minute_scores.csv'):
        rows=cv if s['period']=='CV_pooled' else [r for r in records if r['period']==s['period']]
        count,mae,rmse=score([float(r['target_minutes']) for r in rows],[float(r[s['variant']+'_minutes']) for r in rows])
        require(count==int(s['n']) and math.isclose(mae,float(s['mae']),abs_tol=1e-12) and math.isclose(rmse,float(s['rmse']),abs_tol=1e-12),'Independent minute score mismatch')
    rankings=json.loads((out/'rankings.json').read_text(encoding='utf-8'))
    for period,saved in rankings.items():
        candidates=[r for r in scores if r['period']==period and r['group']=='all']
        expected=sorted(candidates,key=lambda r:(float(r['mae']),float(r['rmse']),len(CASES[r['variant']]['columns']),CASES[r['variant']]['fits'],r['variant']))
        require([r['variant'] for r in saved]==[r['variant'] for r in expected] and [r['rank'] for r in saved]==[1,2,3,4],'Incorrect case ranking')
    pairs=json.loads((out/'paired_intervals.json').read_text(encoding='utf-8'))
    require(len(pairs)==12,'Wrong paired comparison count')
    for period,rows in [('CV_pooled',cv),('February',feb)]:
        comparisons=[p for p in pairs if p['period']==period]
        require({(p['first'],p['second']) for p in comparisons}==set(combinations(CASES,2)),'Missing fixed pair')
        for p in comparisons:
            delta=sum(abs(float(r['target_pts'])-float(r['pred_'+p['second']]))-abs(float(r['target_pts'])-float(r['pred_'+p['first']])) for r in rows)/len(rows)
            require(math.isclose(delta,p['delta_mae'],abs_tol=1e-12) and p['lower_95']<=p['upper_95'],'Paired delta mismatch')
    require(bundles==16 and underlying==28,'Wrong saved model counts')
    return dict(common_feature_rows=n,prediction_rows=len(records),bundles_reloaded=bundles,underlying_models=underlying,
                fits=28,additional_fits=0,case13_exact=True,independent_scores_and_rankings_equal=True,
                cv_best=rankings['CV_pooled'][0]['variant'],february_best=rankings['February'][0]['variant'],
                test_predictions_computed=False)


if __name__=='__main__':
    print(json.dumps(audit(ROOT/'outputs/week2/optimization_combinations_v5')))
