"""Independent raw-sum, context-window, common-cohort and saved-model audit. No fits."""
import json
import math
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from scripts.build_features_baseline import read_csv,sha
from scripts.merge_game_rosters import roster_index
from scripts.opponent_context import build_team_games
from scripts.verify_two_stage import require,score

ROOT=Path(__file__).resolve().parents[1]


def audit(out):
    meta=json.loads((out/'validation.json').read_text(encoding='utf-8'))
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',out)]:
        for p,h in meta[group].items():
            require(sha(base/p)==h,'Changed hash: '+p)
    raw_path=ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv'
    raw=read_csv(raw_path); sums=defaultdict(int); counts=defaultdict(int); opponents={}
    for r in raw:
        day=r['GAMEDATE']
        if day>'2026-02-28': continue
        # Independent team/date sums; deliberately do not use the aggregation helper here.
        pair=r['MATCHUP'].replace(' VS. ',' @ ').split(' @ ')
        require(len(pair)==2 and r['TEAM'] in pair,'Invalid pair')
        opponent=next(t for t in pair if t!=r['TEAM']); key=day,r['TEAM']
        require(key not in opponents or opponents[key]==opponent,'Ambiguous team/day')
        opponents[key]=opponent
        if float(r['MIN'])>0:
            sums[key]+=int(r['PTS']); counts[key]+=1
    teams=read_csv(out/'team_games.csv')
    require(len(teams)==len(sums)==1790,'Wrong team-game count')
    for r in teams:
        key=r['GAMEDATE'],r['team_id']; other=r['GAMEDATE'],r['opponent_id']
        require(opponents[key]==r['opponent_id'],'Wrong opponent identity')
        require(float(r['points_for'])==sums[key] and float(r['points_allowed'])==sums[other],'Wrong raw team score')
        require(float(r['combined_points'])==sums[key]+sums[other],'Wrong combined score')
        require(int(r['positive_minute_players'])==counts[key],'Wrong appeared count')
    rosters=roster_index([r for r in read_csv(raw_path.with_name('nba_2025_26_regular_season_game_rosters.csv')) if r['GAMEDATE']<='2026-02-28'])
    rebuilt=build_team_games(raw,rosters)
    old_index={(r['team_id'],r['game_id']):r for r in teams}
    for r in rebuilt:
        old=old_index[(r['team_id'],r['game_id'])]
        require(all(str(v)==old[k] for k,v in r.items()),'Current aggregation differs from saved result')
    features={}; excluded=read_csv(out/'excluded.csv'); all_old={}; windows_checked=0
    for name in ['train','validation']:
        old=read_csv(ROOT/f'outputs/week2/recent_change_v2/features/{name}.csv')
        all_old.update({(r['player_id'],r['game_id']):r for r in old})
        rows=read_csv(out/f'features/{name}.csv')
        require(len(rows)==(13221 if name=='train' else 3462),'Common cohort length changed')
        for r in rows:
            key=r['player_id'],r['game_id']; require(key not in features,'Duplicate common key'); features[key]=r
    excluded_index={(r['player_id'],r['game_id']):r for r in excluded}
    require(len(excluded)==len(excluded_index)==60,'Wrong excluded cohort')
    require(not (set(features)&set(excluded_index)) and set(features)|set(excluded_index)==set(all_old),'Lost or duplicated original samples')
    for key,r in (features|excluded_index).items():
        old=all_old[key]; require(all(r[k]==v for k,v in old.items()),'Original columns changed')
        day=r['GAMEDATE']; team=r['game_id'].split()[0]
        opponent=next(t for t in r['game_id'].rsplit('|',1)[0].replace(' VS. ',' @ ').split(' @ ') if t!=team)
        require(r['opponent_id']==opponent,'Sample opponent mismatch')
        start=f'{int(day[:4]) if day[5:7]>="07" else int(day[:4])-1}-07-01'
        prior=sorted([r0 for r0 in teams if r0['team_id']==opponent and start<=r0['GAMEDATE']<day],key=lambda r0:r0['GAMEDATE'])
        require(int(r['opponent_history_count'])==len(prior),'Opponent history count mismatch')
        require(json.loads(r['opponent_history_game_ids_5'])==[g['game_id'] for g in prior[-5:]],'Wrong recent five opponent games')
        if key in excluded_index:
            require(len(prior)<5 and r['split']=='train' and r['opponent_exclusion_reason']=='insufficient_opponent_history','Invalid common exclusion')
            continue
        require(len(prior)>=5 and r['opponent_last_history_date']==prior[-1]['GAMEDATE']<day,'Future/insufficient history')
        for column,source in [('opp_points_allowed_mean_5','points_allowed'),('opp_combined_points_mean_5','combined_points')]:
            require(math.isclose(float(r[column]),sum(float(g[source]) for g in prior[-5:])/5,abs_tol=1e-12),'Wrong opponent feature mean')
        windows_checked+=1
    cv=read_csv(out/'predictions_CV_pooled.csv'); feb=read_csv(out/'predictions_February.csv'); records=cv+feb
    require(len(cv)==9023 and len(feb)==3462 and len({(r['player_id'],r['game_id']) for r in records})==12485,'Wrong eval cohort')
    for period,rows,file in [('CV_pooled',cv,'predictions_CV_pooled.csv'),('February',feb,'predictions_February.csv')]:
        controls={(r['player_id'],r['game_id']):r for r in read_csv(ROOT/'outputs/week2/recent_change_v2'/file)}
        require(set(controls)=={(r['player_id'],r['game_id']) for r in rows},'Frozen evaluation cohort changed')
        for r in rows:
            old=controls[(r['player_id'],r['game_id'])]
            for col,source in [('pred_frozen9','pred_v1'),('pred_frozen14','pred_v2'),('target_pts','target_pts'),('pred_baseline','pred_baseline')]:
                require(float(r[col])==float(old[source]),'Frozen control changed')
            require(r['GAMEDATE']<='2026-02-28','Test prediction found')
    fit_log=read_csv(out/'fit_log.csv'); require(len(fit_log)==12,'Wrong fit count')
    require([int(r['fit_number']) for r in fit_log]==list(range(1,13)),'Wrong fit sequence')
    for fit in fit_log:
        rows=[r for r in records if r['period']==fit['period']]
        model=joblib.load(out/'models'/f"{fit['period']}_{fit['variant']}.joblib")
        train=[r for r in features.values() if r['split']=='train' and r['GAMEDATE']<=fit['train_end']]
        require(len(train)==model['n_train']==int(fit['n_train']),'Different training cohort')
        require(model['trained_through']<min(r['GAMEDATE'] for r in rows),'Future-trained model')
        require(model['research_only'],'Missing research status')
        x=np.asarray([[float(features[(r['player_id'],r['game_id'])][c]) for c in model['columns']] for r in rows])
        pred=model['model'].predict(x).astype(float)
        require(np.array_equal(pred,[float(r['pred_'+fit['variant']]) for r in rows]),'Saved prediction differs on reload')
    for s in read_csv(out/'scores.csv'):
        rows=cv if s['period']=='CV_pooled' else [r for r in records if r['period']==s['period']]
        if s['group']!='all': rows=[r for r in rows if float(r['target_pts'])>=int(s['group'][:-1])]
        n,mae,rmse=score([float(r['target_pts']) for r in rows],[float(r['pred_'+s['variant']]) for r in rows])
        require(n==int(s['n']) and math.isclose(mae,float(s['mae']),abs_tol=1e-12) and math.isclose(rmse,float(s['rmse']),abs_tol=1e-12),'Independent score mismatch')
    return dict(team_games=1790,physical_games=895,original_samples_audited=len(all_old),
                eligible_windows_checked=windows_checked,excluded_train_n=60,predictions_checked=12485,
                model_reloads=12,fits=12,additional_fits=0,source_team_sums_equal=True,
                canonical_alias_fix_changes_current_data=False,test_predictions_computed=False)


if __name__=='__main__':
    print(json.dumps(audit(ROOT/'outputs/week2/opponent_context_v4')))
