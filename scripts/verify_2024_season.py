"""No-fit audit of the 2024-25 replication, including independent feature arithmetic."""
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean,pstdev
import joblib
import numpy as np
from scripts.build_features_baseline import read_csv,sha
from scripts.experiment_2024_season import VARIANTS,ALL,PAIRS,OUT,ROOT
from scripts.prior_season_profile import PROFILE_FEATURES,profile_matrix
from scripts.historical_season_features import season_folds
from scripts.high_score_weighting import paired_comparison
from scripts.verify_two_stage import require,score
from scripts.verify_high_score_weighting import close,independent_alerts
from scripts.verify_residual_offset import period_rows


def audit(out=OUT):
    meta=json.loads((out/'validation.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',out)]:
        for p,h in meta[group].items():require(sha(base/p)==h,'Hash changed: '+p)
    data=json.loads((ROOT/'outputs/season_extension_2024_25/data_validation.json').read_text(encoding='utf-8'))
    for group in ['source_sha256','output_sha256','code_sha256']:
        for p,h in data[group].items():require(sha(ROOT/p)==h,'Data artifact changed: '+p)
    original=ROOT/'outputs/nba_2024_25/nba_2024_25_regular_season_player_game_logs.csv'
    require(sha(original)==data['original_2024_csv_sha256'],'Original CSV changed')
    supp=original.parent/'supplement_2026_10_30'
    raw=read_csv(supp/'player_games_enriched.csv');original_rows=read_csv(original)
    key=lambda r:(r['PLAYER'],r['TEAM'],r['GAMEDATE'])
    source={key(r):r for r in original_rows}
    require(len(raw)==len(source)==26306,'Original coverage')
    for r in raw:require({c:r[c] for c in source[key(r)]}==source[key(r)],'Original columns changed')
    # Before touching outcomes, exclude the entire reserved test period.
    dev=[r for r in raw if r['GAMEDATE']<='2025-02-28']
    byplayer=defaultdict(list)
    for r in dev:
        if float(r['MIN'])>0:byplayer[r['player_id']].append(r)
    for rows in byplayer.values():rows.sort(key=lambda r:r['GAMEDATE'])
    teams=defaultdict(list)
    for r in read_csv(supp/'team_games.csv'):
        if r['GAMEDATE']<='2025-02-28':teams[r['team_id']].append(r)
    for rows in teams.values():rows.sort(key=lambda r:r['GAMEDATE'])
    previous=defaultdict(list)
    for r in read_csv(ROOT/'outputs/nba_2023_24/player_games_with_official_ids.csv'):
        require('2023-10-24'<=r['GAMEDATE']<='2024-04-14','Prior date boundary')
        if float(r['MIN'])>0:previous[r['nba_player_id']].append(r)
    clean=read_csv(ROOT/'outputs/nba_2023_24/positive_minute_model_data.csv')
    fields=['nba_player_id','nba_game_id','GAMEDATE','MIN','PTS','REB','AST']
    original_positive=sorted(tuple(r[c] for c in fields) for rows in previous.values() for r in rows)
    cleaned_positive=sorted(tuple(r[c] for c in fields) for r in clean)
    require(all(float(r['MIN'])>0 for r in clean) and original_positive==cleaned_positive,'Clean export changes model inputs')
    names_by_id=defaultdict(set)
    for r in clean:names_by_id[r['nba_player_id']].add(r['PLAYER'])
    require(all(len(names)==1 for names in names_by_id.values()),'Noncanonical clean player names')
    profiles={}
    for pid,rows in previous.items():
        n=len(rows);p=math.fsum(float(r['PTS']) for r in rows);m=math.fsum(float(r['MIN']) for r in rows)
        profiles[pid]=dict(zip(PROFILE_FEATURES,[p/n,m/n,p/m,n,1]))
    saved_profiles=read_csv(out/'previous_profiles.csv');require(len(saved_profiles)==len(profiles),'Prior profile coverage')
    for r in saved_profiles:
        for c,v in profiles[r['nba_player_id']].items():require(close(r[c],v),'Profile arithmetic')
    conference=json.loads((supp/'conferences_2024_25.json').read_text(encoding='utf-8'))
    require(conference['season']=='2024-25' and len(conference['teams'])==30,'Conference snapshot')
    tr=read_csv(out/'features/train.csv');va=read_csv(out/'features/validation.csv');features=tr+va
    feature_map={(r['player_id'],r['game_id']):r for r in features};require(len(feature_map)==len(features),'Duplicate features')
    dev_map={(r['player_id'],r['game_id']):r for r in dev}
    expected_keys=set()
    for r in dev:
        if float(r['MIN'])<=0:continue
        day=r['GAMEDATE'];prior=[h for h in byplayer[r['player_id']] if h['GAMEDATE']<day]
        parts=r['MATCHUP'].replace(' VS. ',' @ ').split(' @ ');opponent=next(t for t in parts if t!=r['TEAM'])
        own=[g for g in teams[r['TEAM']] if g['GAMEDATE']<day];other=[g for g in teams[opponent] if g['GAMEDATE']<day]
        if min(len(prior),len(own),len(other))<5:continue
        k=r['player_id'],r['game_id'];expected_keys.add(k);f=feature_map[k];w5=prior[-5:];w10=prior[-10:]
        expected={name:mean(float(h[c]) for h in window) for name,c,window in [('pts_mean_5','PTS',w5),('pts_mean_10','PTS',w10),('min_mean_5','MIN',w5),('reb_mean_5','REB',w5),('ast_mean_5','AST',w5)]}
        from datetime import date
        expected.update(rest_days=(date.fromisoformat(day)-date.fromisoformat(prior[-1]['GAMEDATE'])).days-1,
            is_home=int(r['is_home']),is_neutral_site=int(r['is_neutral_site']),is_starter=int(r['is_starter']),
            pts_last_1=float(w5[-1]['PTS']),min_mean_3=mean(float(h['MIN']) for h in w5[-3:]),
            min_change_last_vs_5=float(w5[-1]['MIN'])-expected['min_mean_5'],pts_std_5=pstdev(float(h['PTS']) for h in w5),min_std_5=pstdev(float(h['MIN']) for h in w5),
            opp_points_allowed_mean_5=mean(float(h['points_allowed']) for h in other[-5:]),opp_combined_points_mean_5=mean(float(h['combined_points']) for h in other[-5:]))
        same=int(conference['teams'][r['TEAM']]==conference['teams'][opponent]);meetings=[]
        if same:
            for h in prior:
                pair=h['MATCHUP'].replace(' VS. ',' @ ').split(' @ ')
                if next(t for t in pair if t!=h['TEAM'])==opponent:meetings.append(h)
        expected.update(same_conference=same,same_conf_h2h_count=len(meetings),same_conf_h2h_pts_last_1=float(meetings[-1]['PTS']) if meetings else None,
            same_conf_h2h_pts_mean=mean(float(h['PTS']) for h in meetings) if meetings else None)
        expected.update(profiles.get(r['nba_player_id'],dict(zip(PROFILE_FEATURES,[None,None,None,0,0]))))
        for c,v in expected.items():require(close(f[c],v),'Independent feature arithmetic: '+c)
        require(float(f['target_pts'])==float(r['PTS']) and f['last_history_date']<day,'Label/date')
    require(expected_keys==set(feature_map),'Eligibility/shared cohort')
    config=json.loads((out/'experiment_config.json').read_text(encoding='utf-8'))
    require(config['variants']==VARIANTS and config['fit_cap']==8 and config['test_scoring_allowed'] is False,'Frozen candidates')
    tr.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']));folds=season_folds(tr)
    training={n:rows for n,rows,_ in folds};training['February']=tr
    evals={n:{(r['player_id'],r['game_id']) for r in rows} for n,_,rows in folds};evals['February']={(r['player_id'],r['game_id']) for r in va}
    records=[]
    for period in ['CV_pooled','February']:
        rows=read_csv(out/f'predictions_{period}.csv')
        require(all(r['GAMEDATE']<='2025-02-28' for r in rows),'Test predictions appeared')
        records+=rows
    fits=read_csv(out/'fit_log.csv');require(len(fits)==8 and meta['fits']==8,'Fit cap')
    for f in fits:
        v=f['variant'];period=f['period'];spec=VARIANTS[v];train=training[period]
        model=joblib.load(out/'models'/f'{period}_{v}.joblib');require(model['params']==config['params'] and model['columns']==spec['columns'],'Model spec')
        require(all(model['model'].get_params()[k]==x for k,x in config['params'].items()),'Actual parameters')
        high=sum(float(r['target_pts'])>=30 for r in train);norm=1+(spec['multiplier']-1)*high/len(train)
        require(close(f['raw_weight_normalizer'],norm) and math.isclose(model['raw_weight_normalizer'],norm) and int(f['train_high_n'])==high,'Train-only weights')
        rows=[r for r in records if r['period']==period];keys=[(r['player_id'],r['game_id']) for r in rows]
        require(len(keys)==len(set(keys))==len(evals[period]) and set(keys)==evals[period],'Prediction cohort')
        require(model['trained_through']==max(r['GAMEDATE'] for r in train)<min(r['GAMEDATE'] for r in rows),'Temporal fit')
        fs=[feature_map[k] for k in keys]
        pred=model['model'].predict(profile_matrix([{k:x for k,x in r.items() if not k.startswith('target_')} for r in fs],spec['columns'])).astype(float)
        require(np.array_equal(pred,np.asarray([float(r['pred_'+v]) for r in rows])),'Saved predictions')
        for r,feature in zip(rows,fs):require(float(r['target_pts'])==float(feature['target_pts']) and float(r['pred_baseline'])==float(feature['pts_mean_5']),'Scored labels/baseline')
    for s in read_csv(out/'scores.csv'):
        rows=period_rows(records,s['period']);group=s['group']
        if group in ['has_prior','no_prior']:rows=[r for r in rows if int(r['has_prev_season'])==int(group=='has_prior')]
        else:
            filters={'all':lambda y:True,'under30':lambda y:y<30,'20+':lambda y:y>=20,'30+':lambda y:y>=30,
                '0_to_under10':lambda y:y<10,'10_to_under20':lambda y:10<=y<20,'20_to_under30':lambda y:20<=y<30}
            rows=[r for r in rows if filters[group](float(r['target_pts']))]
        require(len(rows)==int(s['n']),'Metric n')
        if not rows:continue
        y=[float(r['target_pts']) for r in rows];p=[float(r['pred_'+s['variant']]) for r in rows];n,mae,rmse=score(y,p)
        for k,v in dict(mae=mae,rmse=rmse,mean_residual=math.fsum(a-b for a,b in zip(y,p))/n,overprediction_fraction=sum(b>a for a,b in zip(y,p))/n).items():require(close(s[k],v),'Metric '+k)
    for s in read_csv(out/'alert_metrics.csv'):
        for k,v in independent_alerts(period_rows(records,s['period']),s['variant']).items():require(close(s[k],v),'Alert/AP '+k)
    intervals=json.loads((out/'paired_intervals.json').read_text(encoding='utf-8'));require(len(intervals)==12,'Interval n')
    for r in intervals:
        again=paired_comparison(period_rows(records,r['period']),r['first'],r['second'],r['group'])
        for k,v in again.items():require(r[k]==v or isinstance(v,float) and math.isclose(r[k],v,rel_tol=0,abs_tol=1e-12),'Interval '+k)
    test=json.loads((out/'test_usage.json').read_text(encoding='utf-8'));require(test['evaluations']==0 and not any(test[k] for k in ['features_built','predictions_computed','metrics_computed']),'Test usage')
    keys=read_csv(out/'test_reserved_keys.csv');require(all(not k.startswith('target_') and k not in ['PTS','MIN','REB','AST'] for k in keys[0]),'Reserved target columns')
    require({(r['player_id'],r['game_id']) for r in keys}=={(r['player_id'],r['game_id']) for r in raw if r['GAMEDATE']>='2025-03-01'},'Reserved test keys')
    return dict(fits=0,saved_models_checked=8,independent_feature_rows=len(features),prior_profiles=len(profiles),
        train_n=len(tr),february_n=len(va),cv_n=len(period_rows(records,'CV_pooled')),source_hashes_verified=True,
        feature_arithmetic_verified=True,metrics_alerts_verified=True,intervals_reproduced_with_shared_bootstrap=True,
        test_evaluations=0,production_changed=False)


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False))
