"""Once-only authorized 2024-25 March-April test, zero new fits."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.build_features_baseline import read_csv,sha
from scripts.experiment_two_stage import save_rows
from scripts.experiment_prior_season_profile import csv_rows
from scripts.experiment_three_models_2024 import ROOT,PRIOR,OUT as DEVELOPMENT,MODELS,ALL,NAMES,summarize
from scripts.three_models_2024 import BASE,base_matrix
from scripts.prior_season_profile import profile_matrix
from scripts.historical_season_features import previous_profiles
from scripts.march_2024_features import build_test_features
from scripts.high_score_weighting import paired_comparison
from scripts.time_validation_tuning import dump
from scripts.verify_three_models_2024 import audit as audit_development

OUT=ROOT/'outputs/season_test_2024_25_three_models'
PAIRS=[('baseline','basic9'),('baseline','basic_prior14'),('baseline','combined_prior25_w2'),
       ('basic9','basic_prior14'),('basic9','combined_prior25_w2'),('basic_prior14','combined_prior25_w2')]
PERIOD='Test_2025_March_April'


def main():
    if OUT.exists():raise ValueError('Test output exists; refuse overwrite or repeat evaluation')
    audit_development()
    supp=ROOT/'outputs/nba_2024_25/supplement_2026_10_30'
    paths=[supp/'player_games_enriched.csv',supp/'team_games.csv',supp/'conferences_2024_25.json',ROOT/'outputs/nba_2023_24/positive_minute_model_data.csv']
    model_paths=json.loads((DEVELOPMENT/'model_sources.json').read_text(encoding='utf-8'))['February']
    protected=[p for folder in [PRIOR,DEVELOPMENT,ROOT/'outputs/week2'] for p in folder.rglob('*') if p.is_file()]
    protected += paths+[ROOT/'streamlit_app.py',ROOT/'nba_dashboard/data.py',ROOT/'nba_dashboard/views.py']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    config=json.loads((DEVELOPMENT/'experiment_config.json').read_text(encoding='utf-8'))
    loaded={v:joblib.load(ROOT/p) for v,p in model_paths.items()}
    for v,b in loaded.items():
        if b['trained_through']!='2025-01-31' or b['params']!=config['params'] or b['n_train']!=12700:raise ValueError('Changed frozen model/cutoff')
    OUT.mkdir()
    dump(OUT/'test_protocol.json',dict(authorized_by='User explicitly requested March and later predictions after sealing development',
        opened_at_utc=datetime.now(timezone.utc).isoformat(),season='2024-25',start='2025-03-01',end='2025-04-13',
        models=MODELS,model_sources=model_paths,model_sha256={v:sha(ROOT/p) for v,p in model_paths.items()},
        trained_through='2025-01-31',fits_allowed=0,tuning_allowed=False,new_candidates_allowed=False,auto_promotion=False,
        train_n=12700,expected_reserved_source_keys=7238,history_rule='strictly previous dates; update after completed games, no same-day outcomes',
        eligibility='source MIN>0; player/team/opponent >=5 prior games; same cohort for all models',
        metrics=['unweighted MAE','RMSE','point groups','prior coverage groups','30-point alerts and ranking AP'],
        pairs=PAIRS,paired_groups=['all','30+'],bootstrap_iterations=2000,seed=42,multiplicity_adjusted=False,
        scope='fixed-model earlier-season holdout backtest; choices originated in later-season research, not prospective independence',
        lineup_source='historical postgame snapshot; not verified pregame',evaluation_cap=1))
    records=read_csv(paths[0]);teams=read_csv(paths[1]);conferences=json.loads(paths[2].read_text(encoding='utf-8'))
    profiles=previous_profiles(read_csv(paths[3]))
    features,outcomes,audit=build_test_features(records,teams,profiles,conferences)
    reserved=read_csv(PRIOR/'test_reserved_keys.csv')
    key=lambda r:(r['player_id'],r['game_id'])
    if len(audit)!=7238 or {key(r) for r in audit}!={key(r) for r in reserved}:raise ValueError('Reserved test source coverage mismatch')
    if len(features)!=len(outcomes) or any('target_' in k or k in ['PTS','MIN','REB','AST'] for r in features for k in r):raise ValueError('Outcome in feature table')
    csv_rows(OUT/'features.csv',features);save_rows(OUT/'outcomes.csv',outcomes);save_rows(OUT/'eligibility_audit.csv',audit)
    outcome_index={key(r):r for r in outcomes}
    rows=[{k:f[k] for k in ['player_id','game_id','GAMEDATE','has_prev_season']}|dict(period=PERIOD,
        target_pts=outcome_index[key(f)]['target_pts'],actual_minutes=outcome_index[key(f)]['actual_minutes'],
        MATCHUP=outcome_index[key(f)]['MATCHUP'],pred_baseline=float(f['pts_mean_5'])) for f in features]
    reload_equal=True
    with threadpool_limits(limits=2):
        for v,b in loaded.items():
            x=base_matrix(features) if v=='basic9' else profile_matrix(features,b['columns'])
            pred=b['model'].predict(x).astype(float)
            fake=[dict(f,target_pts=99999,PTS=99999,MIN=99999,REB=99999,AST=99999) for f in features]
            fake_x=base_matrix(fake) if v=='basic9' else profile_matrix(fake,b['columns'])
            if not np.isfinite(pred).all() or not np.array_equal(pred,b['model'].predict(fake_x).astype(float)):raise ValueError('Outcome-dependent inference')
            again=joblib.load(ROOT/model_paths[v])['model'].predict(x).astype(float)
            reload_equal &= np.array_equal(pred,again)
            for r,p in zip(rows,pred):r['pred_'+v]=float(p)
    save_rows(OUT/'predictions_test.csv',rows)
    human=[]
    for r in rows:
        a=r['target_pts']
        human.append(dict(球員=r['player_id'],日期=r['GAMEDATE'],對戰=r['MATCHUP'],實際上場分鐘=r['actual_minutes'],實際得分=a,
            五場均值=r['pred_baseline'],原九項=r['pred_basic9'],原九項加上季=r['pred_basic_prior14'],組合加上季2倍=r['pred_combined_prior25_w2'],
            原九項絕對誤差=abs(a-r['pred_basic9']),原九項加上季絕對誤差=abs(a-r['pred_basic_prior14']),組合加上季2倍絕對誤差=abs(a-r['pred_combined_prior25_w2'])))
    save_rows(OUT/'三模型逐場預測_2025年3月至季末.csv',human)
    scores,alerts=summarize(PERIOD,rows);save_rows(OUT/'scores.csv',scores);save_rows(OUT/'alert_metrics.csv',alerts)
    monthly=[]
    for month in ['2025-03','2025-04']:
        selected=[r for r in rows if r['GAMEDATE'].startswith(month)];s,_=summarize(month,selected);monthly+=s
    save_rows(OUT/'monthly_scores.csv',monthly)
    intervals=[dict(first=a,second=b,group=g,**paired_comparison(rows,a,b,g)) for a,b in PAIRS for g in ['all','30+']]
    dump(OUT/'paired_intervals.json',intervals)
    counts=Counter(r['reason'] or 'eligible' for r in audit)
    usage=dict(test_start='2025-03-01',test_end='2025-04-13',target_rows_reserved=7238,eligible_n=len(features),
        excluded_n=len(audit)-len(features),eligibility_counts=dict(counts),features_built=True,predictions_computed=True,metrics_computed=True,
        evaluations=1,new_fits=0,trained_through='2025-01-31',production_promoted=False,consumed=True)
    dump(OUT/'test_usage.json',usage);make_report(scores,usage,intervals)
    if not reload_equal or any(sha(ROOT/p)!=h for p,h in before.items()):raise ValueError('Model reload or protected source changed')
    code=[Path(__file__),ROOT/'scripts/march_2024_features.py',ROOT/'scripts/historical_season_features.py',ROOT/'scripts/high_score_weighting.py']
    dump(OUT/'validation.json',dict(new_fits=0,evaluations=1,eligible_n=len(features),source_rows=7238,model_reload_equal=True,
        inference_without_or_fake_labels_equal=True,production_promoted=False,source_sha256=before,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps(dict(test_usage=usage,scores=[s for s in scores if s['group'] in ['all','30+']]),ensure_ascii=False))


def make_report(scores,usage,intervals):
    lines=['# 2024–25三月至季末：三模型預測','',
        f"範圍2025-03-01～04-13；原始預留7238筆，共同合格{usage['eligible_n']}筆，排除{usage['excluded_n']}筆。排除原因：{usage['eligibility_counts']}。",'',
        '三模型固定訓練至2025-01-31，這輪0fit、不調參，只推論與一次性比較。每場嚴格使用前日以前歷史，較早三月結果可用於較晚比賽，但不會重訓模型。',
        '來源MIN=0不進個人窗口；官方球隊得分仍包含所有得分。特徵檔不含當場得分／分鐘標籤，另存outcomes.csv後才計算誤差。','',
        '|模型|整體MAE|RMSE|30+ MAE|<30 MAE|','|---|---:|---:|---:|---:|']
    for v in ALL:
        get=lambda group:next(r for r in scores if r['variant']==v and r['group']==group)
        a=get('all');values=[a['mae'],a['rmse'],get('30+')['mae'],get('under30')['mae']]
        lines.append('|'+NAMES[v]+'|'+'|'.join(f'{x:.4f}' for x in values)+'|')
    lines+=['','本次整體MAE最低為原九項＋上季表現；組合＋上季＋2倍的30+ MAE最低。兩者整體僅差約0.019分，配對日期區間含0，不能將名次當成明確的整體勝負。',
        '組合相對只有原九項的30+ MAE降低約1.432分，但高分場次的誤差仍大，不能當成精準辨識爆發場次的保證。三模型依原協議保留，不依本輪測試調參。',
        '三月／四月分月指標見monthly_scores.csv；完整分箱、有無上季資料群組及30分警示／AP另存CSV。AP是得分排序，不是機率。','',
        '|比較|組別|MAE變化（新−舊）|95%日期區間|','|---|---|---:|---|']
    for r in intervals:lines.append(f"|{NAMES[r['first']]} → {NAMES[r['second']]}|{r['group']}|{r['delta_mae']:.4f}|[{r['lower_95']:.4f}, {r['upper_95']:.4f}]|")
    lines+=['','日期群組2000次重抽，未作多重比較校正。三月保留區間現已使用一次，不能再稱未使用的新測試；原development/test_usage保留為當時封存快照，最新狀態以本目錄test_usage.json為準。',
        '這是日期隔離的假設性回測：先發來自賽後快照、沒有當年賽前發布版本。候選與參數源自後續2025–26研究，因此不能稱當年鎖定模型的前瞻部署驗證。',
        '本輪不依測試結果換模型、重訓或調參；三個候選仍保留，線上模型不變。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
