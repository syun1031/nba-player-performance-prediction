"""Read-only independent arithmetic for the additional consumed-test comparison."""
from collections import defaultdict
from statistics import mean,pstdev
import json
import math
import joblib
import numpy as np
from scripts.build_features_baseline import read_csv,sha
from scripts.predict_prior_2025_test import ROOT,OUT,MODELS,NAMES,MODEL_PATHS,PAIRS
from scripts.experiment_recent_changes import BASE
from scripts.prior_season_profile import PROFILE_FEATURES,profile_matrix
from scripts.verify_prior_season_profile import audit as prior_audit
from scripts.verify_two_stage import require,score
from scripts.verify_high_score_weighting import close,independent_alerts
from scripts.high_score_weighting import paired_comparison

KEY=lambda r:(r['player_id'],r['game_id'])


def audit():
    prior_audit(ROOT/'outputs/week2/prior_season_profile_v9')
    meta=json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',OUT)]:
        for p,h in meta[group].items():require(sha(base/p)==h,'Changed hash '+p)
    protocol=json.loads((OUT/'test_protocol.json').read_text(encoding='utf-8'))
    require(protocol['models']==MODELS and protocol['new_fits_allowed']==0 and protocol['reused_consumed_test'] and not protocol['independent_test'],'Reused-test scope')
    raw=read_csv(ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv')
    raw_index={(r['PLAYER'],r['MATCHUP']+'|'+r['GAMEDATE']):r for r in raw};people=defaultdict(list);totals=defaultdict(int)
    for r in raw:
        totals[r['GAMEDATE'],r['TEAM']]+=int(r['PTS'])
        if float(r['MIN'])>0:people[r['PLAYER']].append(r)
    for rows in people.values():rows.sort(key=lambda r:r['GAMEDATE'])
    team_games=defaultdict(list);seen=set()
    for r in raw:
        key=r['GAMEDATE'],r['TEAM']
        if key in seen:continue
        seen.add(key);pair=r['MATCHUP'].replace(' VS. ',' @ ').split(' @ ');opp=next(t for t in pair if t!=r['TEAM'])
        team_games[r['TEAM']].append(dict(day=r['GAMEDATE'],allowed=totals[r['GAMEDATE'],opp],combined=totals[key]+totals[r['GAMEDATE'],opp]))
    for rows in team_games.values():rows.sort(key=lambda r:r['day'])
    prior=defaultdict(list)
    for r in read_csv(ROOT/'outputs/nba_2024_25/nba_2024_25_regular_season_player_game_logs.csv'):
        require('2024-10-22'<=r['GAMEDATE']<='2025-04-13','Prior dates')
        if float(r['MIN'])>0:prior[r['PLAYER']].append(r)
    profiles={}
    for player,rows in prior.items():
        n=len(rows);points=math.fsum(float(r['PTS']) for r in rows);minutes=math.fsum(float(r['MIN']) for r in rows)
        profiles[player]=dict(zip(PROFILE_FEATURES,[points/n,minutes/n,points/minutes,n,1]))
    targets={KEY(r):r for r in read_csv(ROOT/'outputs/week1/team_lineup_features/test.csv')}
    features=read_csv(OUT/'features.csv');fm={KEY(r):r for r in features}
    require(len(features)==len(fm)==len(targets)==6979 and set(fm)==set(targets),'Exact original cohort')
    require(all(not k.startswith('target_') and k not in ['PTS','MIN','REB','AST'] for k in features[0]),'Outcomes in features')
    conference=json.loads((ROOT/'docs/week1/conferences_2025_26.json').read_text(encoding='utf-8'))['teams']
    for key,f in fm.items():
        r=raw_index[key];day=r['GAMEDATE'];p=[h for h in people[r['PLAYER']] if h['GAMEDATE']<day];w5=p[-5:];w10=p[-10:]
        require('2026-03-01'<=day<='2026-04-12' and float(r['MIN'])>0 and len(p)>=5,'Test dates/eligibility')
        for c in BASE:require(close(f[c],float(targets[key][c])),'Original nine alignment '+c)
        expected={name:mean(float(h[c]) for h in w) for name,c,w in [('pts_mean_5','PTS',w5),('pts_mean_10','PTS',w10),('min_mean_5','MIN',w5),('reb_mean_5','REB',w5),('ast_mean_5','AST',w5)]}
        pair=r['MATCHUP'].replace(' VS. ',' @ ').split(' @ ');opp=next(t for t in pair if t!=r['TEAM']);other=[h for h in team_games[opp] if h['day']<day]
        require(len(other)>=5,'Opponent gate')
        expected.update(pts_last_1=float(w5[-1]['PTS']),min_mean_3=mean(float(h['MIN']) for h in w5[-3:]),
            min_change_last_vs_5=float(w5[-1]['MIN'])-expected['min_mean_5'],pts_std_5=pstdev(float(h['PTS']) for h in w5),min_std_5=pstdev(float(h['MIN']) for h in w5),
            opp_points_allowed_mean_5=mean(h['allowed'] for h in other[-5:]),opp_combined_points_mean_5=mean(h['combined'] for h in other[-5:]))
        same=int(conference[r['TEAM']]==conference[opp]);meetings=[]
        if same:
            for h in p:
                hp=h['MATCHUP'].replace(' VS. ',' @ ').split(' @ ')
                if next(t for t in hp if t!=h['TEAM'])==opp:meetings.append(h)
        expected.update(same_conference=same,same_conf_h2h_count=len(meetings),same_conf_h2h_pts_last_1=float(meetings[-1]['PTS']) if meetings else None,
            same_conf_h2h_pts_mean=mean(float(h['PTS']) for h in meetings) if meetings else None)
        expected.update(profiles.get(r['PLAYER'],dict(zip(PROFILE_FEATURES,[None,None,None,0,0]))))
        for c,v in expected.items():require(close(f[c],v),'Independent feature '+c)
        require(f['last_history_date']==p[-1]['GAMEDATE']<day,'Strict prior date')
    rows=read_csv(OUT/'predictions_test.csv');outcomes={KEY(r):r for r in read_csv(OUT/'outcomes.csv')}
    require(len(rows)==len(outcomes)==len({KEY(r) for r in rows})==6979 and {KEY(r) for r in rows}==set(targets),'Scored keys')
    for r in rows:
        s=raw_index[KEY(r)];require(float(r['target_pts'])==float(s['PTS'])==float(outcomes[KEY(r)]['target_pts']),'Actual labels')
        require(float(r['actual_minutes'])==float(s['MIN']) and float(r['pred_baseline'])==float(fm[KEY(r)]['pts_mean_5']),'Minutes/baseline')
    fs=[fm[KEY(r)] for r in rows]
    for v,path in MODEL_PATHS.items():
        require(sha(ROOT/path)==protocol['model_sha256'][v],'Frozen model SHA')
        b=joblib.load(ROOT/path);require(b['trained_through']=='2026-01-31' and b['n_train']==13221,'Fixed model cohort')
        pred=b['model'].predict(profile_matrix(fs,b['columns'])).astype(float)
        require(np.array_equal(pred,np.asarray([float(r['pred_'+v]) for r in rows])),'Reload prediction')
    filters={'all':lambda y:True,'under30':lambda y:y<30,'20+':lambda y:y>=20,'30+':lambda y:y>=30,'0_to_under10':lambda y:y<10,'10_to_under20':lambda y:10<=y<20,'20_to_under30':lambda y:20<=y<30}
    for s in read_csv(OUT/'scores.csv'):
        g=s['group'];selected=[r for r in rows if int(r['has_prev_season'])==int(g=='has_prior')] if g in ['has_prior','no_prior'] else [r for r in rows if filters[g](float(r['target_pts']))]
        require(len(selected)==int(s['n']),'Score n')
        if not selected:continue
        y=[float(r['target_pts']) for r in selected];p=[float(r['pred_'+s['variant']]) for r in selected];n,mae,rmse=score(y,p)
        for k,v in dict(mae=mae,rmse=rmse,mean_residual=math.fsum(a-b for a,b in zip(y,p))/n,overprediction_fraction=sum(b>a for a,b in zip(y,p))/n).items():require(close(s[k],v),'Score '+k)
    for s in read_csv(OUT/'alert_metrics.csv'):
        for k,v in independent_alerts(rows,s['variant']).items():require(close(s[k],v),'Alert/AP '+k)
    human=read_csv(OUT/'三模型逐場預測_2026年3月至季末.csv');hi={(r['球員'],r['日期']):r for r in human};require(len(hi)==len(human)==6979,'Human keys')
    for r in rows:
        h=hi[r['player_id'],r['GAMEDATE']];actual=float(r['target_pts'])
        require(float(h['實際得分'])==actual and float(h['實際上場分鐘'])==float(r['actual_minutes']),'Human actuals')
        for v in MODELS:require(float(h[NAMES[v]])==float(r['pred_'+v]) and float(h[NAMES[v]+'絕對誤差'])==abs(actual-float(r['pred_'+v])),'Human predictions')
    intervals=json.loads((OUT/'paired_intervals.json').read_text(encoding='utf-8'));require(len(intervals)==12,'Interval count')
    for r in intervals:
        for k,v in paired_comparison(rows,r['first'],r['second'],r['group']).items():require(r[k]==v or isinstance(v,float) and math.isclose(r[k],v,rel_tol=0,abs_tol=1e-12),'Interval '+k)
    return dict(new_fits=0,models_checked=3,independent_feature_rows=6979,original_nine_alignment=True,
        metrics_alerts_verified=True,human_csv_verified=True,hashes_verified=True,interval_bounds_reproduced_with_shared_bootstrap=True,
        reused_consumed_test=True,independent_test=False,production_changed=False)


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False))
