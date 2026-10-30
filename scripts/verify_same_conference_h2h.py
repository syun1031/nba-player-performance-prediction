"""Independent H2H windows, null semantics, scores and saved-model audit. No fits."""
import json
import math
from collections import defaultdict,Counter
from pathlib import Path

import joblib
import numpy as np

from scripts.build_features_baseline import read_csv,sha
from scripts.verify_two_stage import require,score
from scripts.verify_opponent_context import audit as audit_context

ROOT=Path(__file__).resolve().parents[1]


def audit(out):
    meta=json.loads((out/'validation.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',out)]:
        for p,h in meta[group].items(): require(sha(base/p)==h,'Changed hash: '+p)
    context=ROOT/'outputs/week2/opponent_context_v4'; audit_context(context)
    conferences=json.loads((ROOT/'docs/week1/conferences_2025_26.json').read_text(encoding='utf-8'))['teams']
    require(len(conferences)==30 and Counter(conferences.values())=={'East':15,'West':15},'Wrong conference table')
    history=defaultdict(list)
    for r in read_csv(ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv'):
        if r['GAMEDATE']>'2026-02-28': continue
        if float(r['MIN'])<=0: continue
        teams=r['MATCHUP'].replace(' VS. ',' @ ').split(' @ ')
        opponent=next(t for t in teams if t!=r['TEAM'])
        history[(r['PLAYER'],opponent)].append(r)
    features={}; missing_n=0; available_n=0
    for split,size in [('train',13221),('validation',3462)]:
        old={(r['player_id'],r['game_id']):r for r in read_csv(context/f'features/{split}.csv')}
        rows=read_csv(out/f'features/{split}.csv')
        require(len(rows)==len(old)==size,'Changed cohort count')
        require({(r['player_id'],r['game_id']) for r in rows}==set(old),'Changed cohort keys')
        for r in rows:
            key=r['player_id'],r['game_id']; require(all(r[k]==v for k,v in old[key].items()),'Changed old feature values')
            matchup,day=r['game_id'].rsplit('|',1); team=matchup.split()[0]
            pair=matchup.replace(' VS. ',' @ ').split(' @ '); opponent=next(t for t in pair if t!=team)
            same=conferences[team]==conferences[opponent]
            require(int(r['same_conference'])==int(same) and r['h2h_opponent_id']==opponent,'Wrong target conference/opponent')
            prior=sorted([h for h in history[(r['player_id'],opponent)] if '2025-07-01'<=h['GAMEDATE']<day],key=lambda h:h['GAMEDATE']) if same else []
            require(int(r['same_conf_h2h_count'])==len(prior),'Wrong prior meeting count')
            ids=[h['MATCHUP']+'|'+h['GAMEDATE'] for h in prior]
            require(json.loads(r['h2h_history_game_ids'])==ids,'Wrong H2H window or future leakage')
            if prior:
                require(r['h2h_status']=='available' and r['h2h_last_history_date']==prior[-1]['GAMEDATE']<day,'Wrong history status/date')
                require(float(r['same_conf_h2h_pts_last_1'])==float(prior[-1]['PTS']),'Wrong last H2H points')
                require(math.isclose(float(r['same_conf_h2h_pts_mean']),sum(float(h['PTS']) for h in prior)/len(prior),abs_tol=1e-12),'Wrong prior mean')
                available_n+=1
            else:
                require(r['same_conf_h2h_pts_last_1']==r['same_conf_h2h_pts_mean']=='','Absent meetings were filled with a score')
                require(r['h2h_status']==('no_prior_meeting' if same else 'cross_conference'),'Wrong missing status')
                missing_n+=1
            features[key]=r
    cv=read_csv(out/'predictions_CV_pooled.csv'); feb=read_csv(out/'predictions_February.csv'); records=cv+feb
    require(len(cv)==9023 and len(feb)==3462 and len({(r['player_id'],r['game_id']) for r in records})==12485,'Wrong evaluation cohort')
    for rows,file in [(cv,'predictions_CV_pooled.csv'),(feb,'predictions_February.csv')]:
        old={(r['player_id'],r['game_id']):r for r in read_csv(context/file)}
        require(set(old)=={(r['player_id'],r['game_id']) for r in rows},'Changed reference cohort')
        for r in rows:
            key=r['player_id'],r['game_id']; ref=old[key]; feature=features[key]
            for new_col,old_col in [('target_pts','target_pts'),('pred_common9','pred_common9'),('pred_case13','pred_opponent16'),
                                    ('pred_baseline','pred_baseline'),('pred_frozen9','pred_frozen9')]:
                require(float(r[new_col])==float(ref[old_col]),'Changed control/label')
            for col in ['same_conference','same_conf_h2h_count','same_conf_h2h_pts_last_1','same_conf_h2h_pts_mean','h2h_status','h2h_history_game_ids','h2h_last_history_date']:
                require(r[col]==feature[col],'Wrong H2H annotation in predictions')
            require(r['GAMEDATE']<='2026-02-28','Test-period prediction')
    fits=read_csv(out/'fit_log.csv')
    require(len(fits)==8 and [int(r['fit_number']) for r in fits]==list(range(1,9)),'Wrong fit count/sequence')
    config=json.loads((out/'experiment_config.json').read_text(encoding='utf-8'))
    for fit in fits:
        bundle=joblib.load(out/'models'/f"{fit['period']}_{fit['variant']}.joblib")
        rows=[r for r in records if r['period']==fit['period']]
        require(bundle['columns']==config['variants'][fit['variant']] and bundle['research_only'],'Wrong feature whitelist')
        train=[r for r in features.values() if r['split']=='train' and r['GAMEDATE']<=bundle['trained_through']]
        require(len(train)==int(fit['n_train'])==bundle['n_train'],'Wrong train cohort')
        require(bundle['trained_through']<min(r['GAMEDATE'] for r in rows),'Future-trained model')
        # Construct the NaNs independently; never invoke h2h_matrix or pass target fields.
        x=np.asarray([[np.nan if features[(r['player_id'],r['game_id'])][c]=='' else float(features[(r['player_id'],r['game_id'])][c]) for c in bundle['columns']] for r in rows])
        require(bundle['model'].get_booster().num_features()==len(bundle['columns']),'Wrong internal feature count')
        require(np.array_equal(bundle['model'].predict(x).astype(float),[float(r['pred_'+fit['variant']]) for r in rows]),'Reload mismatch')
    for s in read_csv(out/'scores.csv')+read_csv(out/'reference_scores.csv'):
        rows=cv if s['period']=='CV_pooled' else [r for r in records if r['period']==s['period']]
        if s['group']!='all': rows=[r for r in rows if float(r['target_pts'])>=int(s['group'][:-1])]
        n,mae,rmse=score([float(r['target_pts']) for r in rows],[float(r['pred_'+s['variant']]) for r in rows])
        require(n==int(s['n']) and math.isclose(mae,float(s['mae']),abs_tol=1e-12) and math.isclose(rmse,float(s['rmse']),abs_tol=1e-12),'Wrong score')
    for s in read_csv(out/'h2h_group_scores.csv'):
        rows=cv if s['period']=='CV_pooled' else [r for r in records if r['period']==s['period']]
        rows=[r for r in rows if r['h2h_status']==s['h2h_status']]
        n,mae,rmse=score([float(r['target_pts']) for r in rows],[float(r['pred_'+s['variant']]) for r in rows])
        require(n==int(s['n']) and math.isclose(mae,float(s['mae']),abs_tol=1e-12) and math.isclose(rmse,float(s['rmse']),abs_tol=1e-12),'Wrong availability group score')
    for r in read_csv(out/'coverage.csv'):
        rows=[f for f in features.values() if f['split']=='train'] if r['period']=='train' else cv if r['period']=='CV_pooled' else [x for x in records if x['period']==r['period']]
        statuses=Counter(x['h2h_status'] for x in rows); same=statuses['available']+statuses['no_prior_meeting']
        require(int(r['n'])==len(rows) and int(r['same_conference_n'])==same and int(r['with_prior_meeting_n'])==statuses['available'],'Wrong coverage')
        require(int(r['no_prior_meeting_n'])==statuses['no_prior_meeting'] and int(r['cross_conference_n'])==statuses['cross_conference'],'Wrong missing/cross coverage')
        require(math.isclose(float(r['coverage_all']),statuses['available']/len(rows),abs_tol=1e-12),'Wrong overall coverage fraction')
        require(math.isclose(float(r['coverage_within_same']),statuses['available']/same,abs_tol=1e-12),'Wrong coverage fraction')
        histogram=Counter(str(int(x['same_conf_h2h_count'])) for x in rows if int(x['same_conference'])==1)
        require(json.loads(r['same_conference_count_histogram'])==dict(histogram),'Wrong prior-meeting count distribution')
    return dict(feature_rows=len(features),available_feature_rows=available_n,missing_feature_rows=missing_n,
                prediction_rows=len(records),models_reloaded=8,fits=8,additional_fits=0,
                independent_windows_scores_coverage_equal=True,missing_not_zero_verified=True,
                bootstrap_intervals_recomputed=False,test_predictions_computed=False)


if __name__=='__main__':
    print(json.dumps(audit(ROOT/'outputs/week2/same_conference_h2h_v6')))
