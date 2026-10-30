"""Read-only March test audit: independent arithmetic, labels, models and metrics."""
from collections import defaultdict,Counter
from datetime import date
from statistics import mean,pstdev
import json
import math
import joblib
import numpy as np
from scripts.build_features_baseline import read_csv,sha
from scripts.predict_march_2024 import ROOT,OUT,MODELS,PAIRS
from scripts.prior_season_profile import PROFILE_FEATURES,profile_matrix
from scripts.three_models_2024 import BASE,base_matrix
from scripts.verify_three_models_2024 import audit as audit_development
from scripts.verify_two_stage import require,score
from scripts.verify_high_score_weighting import close,independent_alerts
from scripts.high_score_weighting import paired_comparison

KEY=lambda r:(r['player_id'],r['game_id'])


def audit():
    audit_development()
    meta=json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',OUT)]:
        for p,h in meta[group].items():require(sha(base/p)==h,'Hash changed '+p)
    protocol=json.loads((OUT/'test_protocol.json').read_text(encoding='utf-8'))
    require(protocol['models']==MODELS and protocol['fits_allowed']==0 and protocol['trained_through']=='2025-01-31','Fixed protocol')
    supp=ROOT/'outputs/nba_2024_25/supplement_2026_10_30'
    source=read_csv(supp/'player_games_enriched.csv');raw={KEY(r):r for r in source};byplayer=defaultdict(list)
    for r in source:
        if float(r['MIN'])>0:byplayer[r['player_id']].append(r)
    for rows in byplayer.values():rows.sort(key=lambda r:r['GAMEDATE'])
    teams=defaultdict(list)
    for r in read_csv(supp/'team_games.csv'):teams[r['team_id']].append(r)
    for rows in teams.values():rows.sort(key=lambda r:r['GAMEDATE'])
    prior=defaultdict(list)
    for r in read_csv(ROOT/'outputs/nba_2023_24/positive_minute_model_data.csv'):
        require('2023-10-24'<=r['GAMEDATE']<='2024-04-14' and float(r['MIN'])>0,'Prior season')
        prior[r['nba_player_id']].append(r)
    profiles={}
    for pid,rows in prior.items():
        n=len(rows);points=math.fsum(float(r['PTS']) for r in rows);minutes=math.fsum(float(r['MIN']) for r in rows)
        profiles[pid]=dict(zip(PROFILE_FEATURES,[points/n,minutes/n,points/minutes,n,1]))
    conferences=json.loads((supp/'conferences_2024_25.json').read_text(encoding='utf-8'))
    features=read_csv(OUT/'features.csv');feature_map={KEY(r):r for r in features}
    require(len(feature_map)==len(features)==7060,'Feature keys')
    require(all(not c.startswith('target_') and c not in ['PTS','MIN','REB','AST'] for c in features[0]),'Target labels in features')
    eligibility=read_csv(OUT/'eligibility_audit.csv');audit_map={KEY(r):r for r in eligibility}
    reserved={KEY(r) for r in source if '2025-03-01'<=r['GAMEDATE']<='2025-04-13'}
    require(len(eligibility)==len(audit_map)==7238 and set(audit_map)==reserved,'Reserved coverage')
    expected_keys=set();reasons=Counter()
    for key in reserved:
        r=raw[key];day=r['GAMEDATE'];history=[h for h in byplayer[r['player_id']] if h['GAMEDATE']<day]
        pair=r['MATCHUP'].replace(' VS. ',' @ ').split(' @ ');opponent=next(t for t in pair if t!=r['TEAM'])
        own=[g for g in teams[r['TEAM']] if g['GAMEDATE']<day];other=[g for g in teams[opponent] if g['GAMEDATE']<day]
        reason='zero_minutes' if float(r['MIN'])==0 else 'insufficient_history' if len(history)<5 else ''
        require(len(own)>=5 and len(other)>=5,'Unexpected team gate in test')
        require(audit_map[key]['reason']==reason and audit_map[key]['eligible']==str(not reason),'Eligibility reason')
        reasons[reason or 'eligible']+=1
        if reason:continue
        expected_keys.add(key);f=feature_map[key];w5=history[-5:];w10=history[-10:]
        expected={name:mean(float(h[c]) for h in window) for name,c,window in [('pts_mean_5','PTS',w5),('pts_mean_10','PTS',w10),('min_mean_5','MIN',w5),('reb_mean_5','REB',w5),('ast_mean_5','AST',w5)]}
        starters=json.loads(r['starting_players'])
        expected.update(rest_days=(date.fromisoformat(day)-date.fromisoformat(history[-1]['GAMEDATE'])).days-1,
            is_home=int(r['is_home']),is_neutral_site=int(r['is_neutral_site']),is_starter=int(r['player_id'] in starters),
            pts_last_1=float(w5[-1]['PTS']),min_mean_3=mean(float(h['MIN']) for h in w5[-3:]),
            min_change_last_vs_5=float(w5[-1]['MIN'])-expected['min_mean_5'],pts_std_5=pstdev(float(h['PTS']) for h in w5),min_std_5=pstdev(float(h['MIN']) for h in w5),
            opp_points_allowed_mean_5=mean(float(h['points_allowed']) for h in other[-5:]),opp_combined_points_mean_5=mean(float(h['combined_points']) for h in other[-5:]))
        same=int(conferences['teams'][r['TEAM']]==conferences['teams'][opponent]);meetings=[]
        if same:
            for h in history:
                hp=h['MATCHUP'].replace(' VS. ',' @ ').split(' @ ')
                if next(t for t in hp if t!=h['TEAM'])==opponent:meetings.append(h)
        expected.update(same_conference=same,same_conf_h2h_count=len(meetings),same_conf_h2h_pts_last_1=float(meetings[-1]['PTS']) if meetings else None,
            same_conf_h2h_pts_mean=mean(float(h['PTS']) for h in meetings) if meetings else None)
        expected.update(profiles.get(r['nba_player_id'],dict(zip(PROFILE_FEATURES,[None,None,None,0,0]))))
        for c,v in expected.items():require(close(f[c],v),'Independent test feature '+c)
        require(f['last_history_date']==history[-1]['GAMEDATE']<day and f['split']=='test','Prior-date history')
        require(json.loads(f['history_game_ids_5'])==[h['game_id'] for h in w5],'Prior-five keys')
        require(json.loads(f['opponent_history_game_ids_5'])==[h['game_id'] for h in other[-5:]],'Opponent prior-five keys')
    require(expected_keys==set(feature_map),'Common eligible cohort')
    outcomes=read_csv(OUT/'outcomes.csv');outcome_map={KEY(r):r for r in outcomes}
    predictions=read_csv(OUT/'predictions_test.csv')
    require(len(outcomes)==len(predictions)==len({KEY(r) for r in predictions})==7060 and set(outcome_map)==expected_keys,'Outcome/prediction cohort')
    for r in predictions:
        s=raw[KEY(r)];o=outcome_map[KEY(r)]
        require(float(r['target_pts'])==float(o['target_pts'])==float(s['PTS']) and float(r['actual_minutes'])==float(s['actual_minutes']),'Actual outcomes')
        require(float(r['pred_baseline'])==float(feature_map[KEY(r)]['pts_mean_5']),'Baseline')
    fs=[feature_map[KEY(r)] for r in predictions]
    for v,path in protocol['model_sources'].items():
        require(sha(ROOT/path)==protocol['model_sha256'][v],'Frozen model hash')
        b=joblib.load(ROOT/path);require(b['trained_through']=='2025-01-31' and b['n_train']==12700,'Frozen cutoff')
        if v=='basic9':require(b['columns']==BASE,'Nine-feature whitelist')
        x=base_matrix(fs) if v=='basic9' else profile_matrix(fs,b['columns'])
        require(np.array_equal(b['model'].predict(x).astype(float),np.asarray([float(r['pred_'+v]) for r in predictions])),'Frozen reload prediction')
    filters={'all':lambda y:True,'under30':lambda y:y<30,'20+':lambda y:y>=20,'30+':lambda y:y>=30,'0_to_under10':lambda y:y<10,'10_to_under20':lambda y:10<=y<20,'20_to_under30':lambda y:20<=y<30}
    for file in ['scores.csv','monthly_scores.csv']:
        for s in read_csv(OUT/file):
            rows=predictions if file=='scores.csv' else [r for r in predictions if r['GAMEDATE'].startswith(s['period'])]
            g=s['group'];rows=[r for r in rows if int(r['has_prev_season'])==int(g=='has_prior')] if g in ['has_prior','no_prior'] else [r for r in rows if filters[g](float(r['target_pts']))]
            require(len(rows)==int(s['n']),'Metric n')
            if not rows:continue
            y=[float(r['target_pts']) for r in rows];p=[float(r['pred_'+s['variant']]) for r in rows];n,mae,rmse=score(y,p)
            for k,v in dict(mae=mae,rmse=rmse,mean_residual=math.fsum(a-b for a,b in zip(y,p))/n,overprediction_fraction=sum(b>a for a,b in zip(y,p))/n).items():require(close(s[k],v),'Metric '+k)
    for s in read_csv(OUT/'alert_metrics.csv'):
        for k,v in independent_alerts(predictions,s['variant']).items():require(close(s[k],v),'Alert/AP '+k)
    human=read_csv(OUT/'三模型逐場預測_2025年3月至季末.csv');hi={(r['球員'],r['日期']):r for r in human};require(len(human)==len(hi)==7060,'Human keys')
    for r in predictions:
        h=hi[r['player_id'],r['GAMEDATE']];actual=float(r['target_pts'])
        require(float(h['實際得分'])==actual and float(h['實際上場分鐘'])==float(r['actual_minutes']) and h['對戰']==r['MATCHUP'],'Human outcomes')
        for v,label,error in [('basic9','原九項','原九項絕對誤差'),('basic_prior14','原九項加上季','原九項加上季絕對誤差'),('combined_prior25_w2','組合加上季2倍','組合加上季2倍絕對誤差')]:
            p=float(r['pred_'+v]);require(float(h[label])==p and float(h[error])==abs(actual-p),'Human forecast/error')
    intervals=json.loads((OUT/'paired_intervals.json').read_text(encoding='utf-8'));require(len(intervals)==12,'Interval count')
    for r in intervals:
        require((r['first'],r['second']) in PAIRS,'Comparison pair')
        for k,v in paired_comparison(predictions,r['first'],r['second'],r['group']).items():require(r[k]==v or isinstance(v,float) and math.isclose(r[k],v,rel_tol=0,abs_tol=1e-12),'Interval '+k)
    usage=json.loads((OUT/'test_usage.json').read_text(encoding='utf-8'))
    require(usage['evaluations']==1 and usage['new_fits']==0 and usage['consumed'] and usage['eligibility_counts']==dict(reasons),'Consumed test ledger')
    return dict(new_fits=0,models_checked=3,source_test_rows=7238,independent_feature_rows=7060,excluded_n=178,
        eligibility_counts=dict(reasons),metrics_alerts_verified=True,human_csv_verified=True,hashes_verified=True,
        interval_bounds_reproduced_with_shared_bootstrap=True,test_evaluations=1,production_changed=False)


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False))
