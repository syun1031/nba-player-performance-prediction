"""Additional fixed-model comparison on the already consumed 2025-26 test."""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.build_features_baseline import read_csv,sha
from scripts.experiment_two_stage import save_rows
from scripts.experiment_prior_season_profile import csv_rows
from scripts.experiment_three_models_2024 import MODELS,ALL,NAMES,summarize
from scripts.prior_season_profile import build_profiles,PROFILE_FEATURES,profile_matrix
from scripts.same_conference_h2h import h2h_lineup_features
from scripts.merge_game_rosters import roster_index
from scripts.luka_weighted_demo import build_prior_team_games
from scripts.team_lineup_features import TeamHistoryIndex
from scripts.experiment_recent_changes import BASE
from scripts.high_score_weighting import paired_comparison
from scripts.time_validation_tuning import dump
from scripts.verify_prior_season_profile import audit as audit_prior

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/season_test_2025_26_prior_models'
PAIRS=[('baseline','basic9'),('baseline','basic_prior14'),('baseline','combined_prior25_w2'),('basic9','basic_prior14'),('basic9','combined_prior25_w2'),('basic_prior14','combined_prior25_w2')]
PERIOD='Reused_test_2026_March_April'
MODEL_PATHS={'basic9':'outputs/week2/opponent_context_v4/models/February_common9.joblib',
    'basic_prior14':'outputs/week2/prior_season_profile_v9/models/February_basic_prior14.joblib',
    'combined_prior25_w2':'outputs/week2/prior_season_profile_v9/models/February_combined_prior25_w2.joblib'}


def build_features(raw,roster_rows,targets,profiles):
    rosters=roster_index(roster_rows)
    teams=TeamHistoryIndex(build_prior_team_games(raw,rosters,'2026-04-13'))
    personal=defaultdict(list)
    for r in raw:
        personal[r['PLAYER']].append(dict(player_id=r['PLAYER'],team_id=r['TEAM'],game_id=r['MATCHUP']+'|'+r['GAMEDATE'],
            GAMEDATE=r['GAMEDATE'],**{c:r[c] for c in ['MIN','PTS','REB','AST']}))
    raw_index={(r['PLAYER'],r['MATCHUP']+'|'+r['GAMEDATE']):r for r in raw}
    features=[];outcomes=[]
    for r in sorted(targets,key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id'])):
        if not '2026-03-01'<=r['GAMEDATE']<='2026-04-12':raise ValueError('Only authorized 2025-26 test')
        target={k:v for k,v in r.items() if not k.startswith('target_')};target['team_id']=r['game_id'].split()[0]
        starters=rosters[r['GAMEDATE'],target['team_id']]['starters']
        f=h2h_lineup_features(personal[r['player_id']],teams,target,starters,mode='historical_postgame')
        if not f['eligible'] or f['live_prediction_ready']:raise ValueError('Changed common test eligibility')
        for c in BASE:
            if not np.isclose(float(f[c]),float(r[c]),rtol=0,atol=1e-12):raise ValueError('Original nine feature mismatch: '+c)
        f.update(profiles.get(r['player_id'],dict(zip(PROFILE_FEATURES,[np.nan,np.nan,np.nan,0,0]))))
        ids={k:r[k] for k in ['player_id','game_id','GAMEDATE']}
        features.append(dict(ids,team_id=target['team_id'],season='2025-26',split='test',**f))
        actual=raw_index[r['player_id'],r['game_id']]
        if float(actual['MIN'])<=0 or float(actual['PTS'])!=float(r['target_pts']):raise ValueError('Outcome/appearance mismatch')
        outcomes.append(dict(ids,target_pts=float(actual['PTS']),actual_minutes=float(actual['MIN']),MATCHUP=actual['MATCHUP']))
    return features,outcomes


def main():
    if OUT.exists():raise ValueError('Existing reused-test output cannot be overwritten')
    audit_prior(ROOT/'outputs/week2/prior_season_profile_v9')
    rawpath=ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv';rosterpath=rawpath.with_name('nba_2025_26_regular_season_game_rosters.csv')
    targetpath=ROOT/'outputs/week1/team_lineup_features/test.csv';priorpath=ROOT/'outputs/nba_2024_25/nba_2024_25_regular_season_player_game_logs.csv'
    protected=[p for p in (ROOT/'outputs/week2').rglob('*') if p.is_file()]+[rawpath,rosterpath,targetpath,priorpath,ROOT/'docs/week1/conferences_2025_26.json']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    bundles={v:joblib.load(ROOT/path) for v,path in MODEL_PATHS.items()}
    for v,b in bundles.items():
        if b['trained_through']!='2026-01-31' or b['n_train']!=13221:raise ValueError('Frozen model cutoff/cohort mismatch')
    OUT.mkdir()
    dump(OUT/'test_protocol.json',dict(authorized_by='User requested 2025-26 prior-profile models too',opened_at_utc=datetime.now(timezone.utc).isoformat(),
        season='2025-26',start='2026-03-01',end='2026-04-12',source_n=6979,models=MODELS,model_sources=MODEL_PATHS,
        model_sha256={v:sha(ROOT/p) for v,p in MODEL_PATHS.items()},trained_through='2026-01-31',n_train=13221,new_fits_allowed=0,
        tuning_allowed=False,auto_promotion=False,independent_test=False,reused_consumed_test=True,
        history_rule='strictly previous date; prior completed test games may update later windows without fitting',
        prior_profile_rule='same exact PLAYER join and missing handling as original v9 training; no new ID re-encoding',
        pairs=PAIRS,bootstrap_iterations=2000,seed=42,multiplicity_adjusted=False,original_protocols_unchanged=True))
    targets=read_csv(targetpath)
    if len(targets)!=6979 or len({(r['player_id'],r['game_id']) for r in targets})!=6979:raise ValueError('Changed original test cohort')
    features,outcomes=build_features(read_csv(rawpath),read_csv(rosterpath),targets,build_profiles(read_csv(priorpath)))
    csv_rows(OUT/'features.csv',features);save_rows(OUT/'outcomes.csv',outcomes)
    oi={(r['player_id'],r['game_id']):r for r in outcomes}
    rows=[{k:r[k] for k in ['player_id','game_id','GAMEDATE','has_prev_season']}|dict(period=PERIOD,
        target_pts=oi[r['player_id'],r['game_id']]['target_pts'],actual_minutes=oi[r['player_id'],r['game_id']]['actual_minutes'],
        MATCHUP=oi[r['player_id'],r['game_id']]['MATCHUP'],pred_baseline=float(r['pts_mean_5'])) for r in features]
    with threadpool_limits(limits=2):
        for v,b in bundles.items():
            x=profile_matrix(features,b['columns']);pred=b['model'].predict(x).astype(float)
            fake=profile_matrix([dict(r,target_pts=99999,PTS=99999,MIN=99999) for r in features],b['columns'])
            if not np.isfinite(pred).all() or not np.array_equal(pred,b['model'].predict(fake).astype(float)) or not np.array_equal(pred,joblib.load(ROOT/MODEL_PATHS[v])['model'].predict(x).astype(float)):raise ValueError('Prediction/reload/outcome dependence')
            for r,p in zip(rows,pred):r['pred_'+v]=float(p)
    save_rows(OUT/'predictions_test.csv',rows)
    human=[]
    for r in rows:
        h=dict(球員=r['player_id'],日期=r['GAMEDATE'],對戰=r['MATCHUP'],實際上場分鐘=r['actual_minutes'],實際得分=r['target_pts'],五場均值=r['pred_baseline'])
        for v in MODELS:h[NAMES[v]]=r['pred_'+v];h[NAMES[v]+'絕對誤差']=abs(r['target_pts']-r['pred_'+v])
        human.append(h)
    save_rows(OUT/'三模型逐場預測_2026年3月至季末.csv',human)
    scores,alerts=summarize(PERIOD,rows);save_rows(OUT/'scores.csv',scores);save_rows(OUT/'alert_metrics.csv',alerts)
    intervals=[dict(first=a,second=b,group=g,**paired_comparison(rows,a,b,g)) for a,b in PAIRS for g in ['all','30+']]
    dump(OUT/'paired_intervals.json',intervals)
    dump(OUT/'test_usage.json',dict(start='2026-03-01',end='2026-04-12',n=6979,new_fits=0,new_independent_test=False,
        reused_consumed_test=True,additional_model_comparisons=1,old_test_usage_unchanged=True,production_promoted=False))
    lines=['# 2025–26上季表現模型：三月至季末追加比較','',
        '這是使用者要求在已使用過的6979筆測試資料上追加固定模型比較，不是新的獨立最終測試。0fit、不調參，全部固定訓練至2026-01-31。原九項為同13221訓練cohort對照，與部署主模型的13281訓練樣本不同。','',
        '|模型|整體MAE|RMSE|30+ MAE|','|---|---:|---:|---:|']
    for v in ALL:
        a=next(s for s in scores if s['variant']==v and s['group']=='all');h=next(s for s in scores if s['variant']==v and s['group']=='30+')
        lines.append(f"|{NAMES[v]}|{a['mae']:.4f}|{a['rmse']:.4f}|{h['mae']:.4f}|")
    lines+=['','完整分箱／有無上季資料分組、30分警示與AP、12個日期區間另存CSV/JSON；區間未作多重比較校正。',
        '全部原九項窗口與原test逐值一致；新增特徵只用嚴格前日歷史，先發仍是賽後快照，屬日期隔離的假設性回測。上季資料依訓練時完全姓名接合，沒有改接合政策或模型。',
        '舊測試協議與模型均不變；這輪結果不作追加調參、換模或獨立前瞻部署證據。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if any(sha(ROOT/p)!=h for p,h in before.items()):raise ValueError('Protected research source changed')
    dump(OUT/'validation.json',dict(new_fits=0,source_n=6979,inference_without_or_fake_labels_equal=True,model_reload_equal=True,
        independent_test=False,reused_consumed_test=True,production_promoted=False,source_sha256=before,
        code_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
        output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps([s for s in scores if s['group'] in ['all','30+']],ensure_ascii=False))


if __name__=='__main__':main()
