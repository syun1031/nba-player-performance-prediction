"""Read-only audit of saved two-stage artifacts, with zero fits."""
import json
import math
from pathlib import Path

import joblib
import numpy as np

from scripts.build_features_baseline import read_csv,sha
from scripts.two_stage_model import FEATURES

ROOT=Path(__file__).resolve().parents[1]


def require(condition,message):
    if not condition:
        raise ValueError(message)


def score(y,p):
    errors=[a-b for a,b in zip(y,p)]
    return len(errors),sum(abs(e) for e in errors)/len(errors),math.sqrt(sum(e*e for e in errors)/len(errors))


def audit(out):
    meta=json.loads((out/'validation.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',out)]:
        for p,h in meta[group].items():
            require(sha(base/p)==h,'Hash changed: '+p)
    source={}
    for r in read_csv(ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv'):
        if r['GAMEDATE']>'2026-02-28':
            continue
        source[(r['PLAYER'],r['MATCHUP']+'|'+r['GAMEDATE'])]=r
    features={}; audited_n=0
    for name,expected in [('train',13281),('validation',3462)]:
        old=read_csv(ROOT/f'outputs/week2/recent_change_v2/features/{name}.csv')
        rows=read_csv(out/f'features/{name}.csv')
        index={(r['player_id'],r['game_id']):r for r in rows}
        require(len(rows)==len(index)==len(old)==expected,'Feature cohort changed')
        for r in old:
            key=r['player_id'],r['game_id']; new=index[key]; raw=source[key]
            require(all(new[c]==v for c,v in r.items()),'Original development values changed')
            require(float(raw['MIN'])==float(new['target_minutes'])>0,'Minute label mismatch')
            require(float(raw['PTS'])==float(new['target_pts']),'Point label mismatch')
            require(math.isclose(float(new['target_ppm']),float(raw['PTS'])/float(raw['MIN']),abs_tol=1e-12),'Rate label mismatch')
            require(new['last_history_date']<new['GAMEDATE'],'History leakage')
            for n in [3,5,10]:
                ids=json.loads(new[f'history_game_ids_{n}'])
                require(all(i.rsplit('|',1)[1]<new['GAMEDATE'] for i in ids),'Window leakage')
            features[key]=new; audited_n+=1
    records=read_csv(out/'predictions_CV_pooled.csv')+read_csv(out/'predictions_February.csv')
    require(len(records)==12485 and len({(r['player_id'],r['game_id']) for r in records})==12485,'Prediction cohort mismatch')
    control_files=[('CV_pooled','outputs/week2/recent_change_v2/predictions_CV_pooled.csv'),
                   ('February','outputs/week2/recent_change_v2/predictions_February.csv')]
    controls={}
    for _,file in control_files:
        controls.update({(r['player_id'],r['game_id']):r for r in read_csv(ROOT/file)})
    for r in records:
        key=r['player_id'],r['game_id']; source_row=features[key]; control=controls[key]
        require(r['GAMEDATE']<='2026-02-28','Test-period prediction found')
        for c in ['target_pts','pred_v1','pred_v2','pred_baseline']:
            require(float(r[c])==float(control[c]),'Frozen control mismatch')
        require(float(r['target_minutes'])==float(source_row['target_minutes']),'Wrong minutes in scored row')
        require(float(r['pred_two_stage'])==float(r['pred_minutes'])*float(r['pred_rate']),'Product mismatch')
        require(float(r['diagnostic_actual_minutes_points'])==float(r['target_minutes'])*float(r['pred_rate']),'Diagnostic product mismatch')
    fits=read_csv(out/'fit_log.csv')
    require(len(fits)==8 and [int(r['fit_number']) for r in fits]==list(range(1,9)),'Fit budget mismatch')
    for period in ['F1','F2','F3','February']:
        rows=[r for r in records if r['period']==period]
        model=joblib.load(out/'models'/f'{period}.joblib')
        require(model.columns==FEATURES and model.research_only,'Wrong saved model specification')
        train=[r for r in features.values() if r['split']=='train' and r['GAMEDATE']<=model.trained_through]
        require(len(train)==model.training_n,'Model training count mismatch')
        require(math.isclose(model.weight_normalizer,sum(float(r['target_minutes']) for r in train)/len(train),abs_tol=1e-12),'Training-only weight mismatch')
        require(model.trained_through<min(r['GAMEDATE'] for r in rows),'Future-trained model')
        x=np.asarray([[float(features[(r['player_id'],r['game_id'])][c]) for c in FEATURES] for r in rows])
        minutes=np.maximum(model.minutes_model.predict(x).astype(float),0)
        rate=np.maximum(model.rate_model.predict(x).astype(float),0)
        require(np.array_equal(minutes,[float(r['pred_minutes']) for r in rows]),'Reloaded minutes changed')
        require(np.array_equal(rate,[float(r['pred_rate']) for r in rows]),'Reloaded rate changed')
    for s in read_csv(out/'scores.csv'):
        rows=records if s['period']=='CV_pooled' else [r for r in records if r['period']==s['period']]
        if s['period']=='CV_pooled':
            rows=[r for r in rows if r['period']!='February']
        if s['group']!='all':
            rows=[r for r in rows if float(r['target_pts'])>=int(s['group'][:-1])]
        n,mae,rmse=score([float(r['target_pts']) for r in rows],[float(r['pred_'+s['variant']]) for r in rows])
        require(n==int(s['n']) and math.isclose(mae,float(s['mae']),abs_tol=1e-12) and math.isclose(rmse,float(s['rmse']),abs_tol=1e-12),'Score mismatch')
    for s in read_csv(out/'minute_scores.csv'):
        rows=[r for r in records if r['period']!='February'] if s['period']=='CV_pooled' else [r for r in records if r['period']==s['period']]
        col='pred_minutes' if s['variant']=='predicted_minutes' else 'min_mean_5'
        n,mae,rmse=score([float(r['target_minutes']) for r in rows],[float(r[col]) for r in rows])
        require(n==int(s['n']) and math.isclose(mae,float(s['mae']),abs_tol=1e-12) and math.isclose(rmse,float(s['rmse']),abs_tol=1e-12),'Minute score mismatch')
    return dict(feature_rows=audited_n,prediction_rows=len(records),fit_count=8,
                exact_controls=True,reloaded_components_equal=True,independent_scores_equal=True,
                additional_fits=0,test_predictions_computed=False)


if __name__=='__main__':
    print(json.dumps(audit(ROOT/'outputs/week2/two_stage_v3')))
