"""No-fit comparison: two saved mean-offset models versus two prior-season models."""
import json
import math
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.build_features_baseline import read_csv,sha
from scripts.experiment_two_stage import save_rows
from scripts.high_score_weighting import point_group_scores,alert_metrics,paired_comparison
from scripts.time_validation_tuning import dump
from scripts.verify_residual_offset import audit as audit_offset
from scripts.verify_prior_2025_test import audit as audit_prior_test

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/offset_vs_prior_2025_26'
OFFSET=ROOT/'outputs/week2/residual_offset_v8'
PRIOR=ROOT/'outputs/week2/prior_season_profile_v9'
TEST=ROOT/'outputs/season_test_2025_26_prior_models'
MODELS=['residual_w1','residual_w2','basic_prior14','combined_prior25_w2']
ALL=['baseline']+MODELS
PAIRS=[(a,b) for a in MODELS[:2] for b in MODELS[2:]]
NAMES=dict(baseline='五場均值',residual_w1='五場均值＋增減（未加權）',residual_w2='五場均值＋增減（高分2倍）',
    basic_prior14='原九項＋上季表現',combined_prior25_w2='1＋3＋同區＋上季表現＋2倍')
KEY=lambda r:(r['player_id'],r['game_id'])


def align_controls(first,second):
    a={KEY(r):r for r in first};b={KEY(r):r for r in second}
    if not a or len(a)!=len(first) or len(b)!=len(second) or set(a)!=set(b):raise ValueError('Different or duplicate common cohort')
    for r in first:
        s=b[KEY(r)]
        if r['GAMEDATE']!=s['GAMEDATE'] or r['period']!=s['period'] or r['game_id'].rsplit('|',1)[-1]!=r['GAMEDATE']:raise ValueError('Different date or fold')
        for c in ['target_pts','pred_baseline']:
            x,y=float(r[c]),float(s[c])
            if not math.isfinite(x) or x<0 or x!=y:raise ValueError('Different/invalid label or baseline')
    return [(r,b[KEY(r)]) for r in first]


def summarize(period,rows):
    scores=[];alerts=[]
    for v in ALL:
        y=[float(r['target_pts']) for r in rows];p=[float(r['pred_'+v]) for r in rows]
        scores.extend(dict(dataset=period,variant=v,**s) for s in point_group_scores(y,p))
        for flag,group in [(1,'has_prior'),(0,'no_prior')]:
            selected=[r for r in rows if int(r['has_prev_season'])==flag]
            if selected:
                s=point_group_scores([float(r['target_pts']) for r in selected],[float(r['pred_'+v]) for r in selected])[0]
                scores.append(dict(dataset=period,variant=v,**dict(s,group=group)))
        alerts.append(dict(dataset=period,variant=v,**alert_metrics(y,p)))
    return scores,alerts


def main():
    if OUT.exists():raise ValueError('Refuse existing output or repeated evaluation')
    audit_offset(OFFSET);audit_prior_test()
    paths=[OFFSET/'models'/f'February_{v}.joblib' for v in MODELS[:2]]
    previous=[p for folder in [OFFSET,PRIOR,TEST] for p in folder.rglob('*') if p.is_file()]
    rawpath=ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv'
    previous += [rawpath,ROOT/'nba_dashboard/data.py',ROOT/'nba_dashboard/views.py',ROOT/'nba_dashboard/research_views.py',ROOT/'dashboard_pages/comparison.py']
    before={str(p.relative_to(ROOT)):sha(p) for p in previous}
    models={v:joblib.load(p) for v,p in zip(MODELS[:2],paths)}
    prior_models={v:joblib.load(PRIOR/'models'/f'February_{v}.joblib') for v in MODELS[2:]}
    for v,m in models.items():
        if m.training_n!=13221 or m.trained_through!='2026-01-31' or m.params!=prior_models['basic_prior14']['params']:raise ValueError('Different frozen training configuration')
    OUT.mkdir()
    dump(OUT/'comparison_protocol.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),season='2025-26',models=MODELS,
        periods={'CV_pooled':9023,'February':3462,'Test':6979},new_fits_allowed=0,tuning_allowed=False,production_promoted=False,
        train_n=13221,trained_through='2026-01-31',cv_note='same expanding folds, with their earlier cutoffs',
        offset_features=20,prior_feature_counts=[14,25],comparison_scope='whole model schemes; not a controlled estimate of offset target alone',
        test_scope='additional research on already consumed test, not new independent final test',
        model_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths+[PRIOR/'models'/f'February_{v}.joblib' for v in MODELS[2:]]},
        pairs=PAIRS,paired_groups=['all','30+'],bootstrap_iterations=2000,seed=42,multiplicity_adjusted=False))
    test_controls=read_csv(TEST/'predictions_test.csv');features=read_csv(TEST/'features.csv');fi={KEY(r):r for r in features}
    if len(fi)!=len(features) or len(features)!=6979:raise ValueError('Invalid frozen test features')
    if {KEY(r) for r in test_controls}!=set(fi) or len(test_controls)!=6979:raise ValueError('Different test feature cohort')
    component={}
    with threadpool_limits(limits=2):
        for v,m in models.items():
            fs=[fi[KEY(r)] for r in test_controls];result=m.predict_components(fs)
            altered=m.predict_components([dict(r,target_pts=99999,PTS=99999,MIN=99999) for r in fs])
            if any(not np.array_equal(result[k],altered[k]) for k in result):raise ValueError('Outcome-dependent offset inference')
            again=joblib.load(OFFSET/'models'/f'February_{v}.joblib').predict_components(fs)
            if any(not np.array_equal(result[k],again[k]) for k in result):raise ValueError('Offset reload mismatch')
            component[v]=result
    residual=[]
    for i,r in enumerate(test_controls):
        s={k:r[k] for k in ['player_id','game_id','GAMEDATE','target_pts','period','pred_baseline']}
        for v in MODELS[:2]:
            for field,source in [('pred_'+v,'points'),(v+'_delta','delta'),(v+'_raw_points','raw_points')]:s[field]=float(component[v][source][i])
        residual.append(s)
    datasets={};scores=[];alerts=[];intervals=[]
    raw={(r['PLAYER'],r['MATCHUP']+'|'+r['GAMEDATE']):r for r in read_csv(rawpath)}
    for dataset in ['CV_pooled','February','Test']:
        left=residual if dataset=='Test' else read_csv(OFFSET/f'predictions_{dataset}.csv')
        right=test_controls if dataset=='Test' else read_csv(PRIOR/f'predictions_{dataset}.csv')
        aligned=align_controls(left,right);rows=[]
        for a,b in aligned:
            r={k:a[k] for k in ['player_id','game_id','GAMEDATE','target_pts','period','pred_baseline']}
            r['has_prev_season']=int(b['has_prev_season'])
            for v in MODELS[:2]:
                for c in ['pred_'+v,v+'_delta',v+'_raw_points']:r[c]=float(a[c])
            for v in MODELS[2:]:r['pred_'+v]=float(b['pred_'+v])
            if not np.isfinite([r['pred_'+v] for v in MODELS]).all():raise ValueError('Nonfinite comparison prediction')
            rows.append(r)
        if len(rows)!={'CV_pooled':9023,'February':3462,'Test':6979}[dataset]:raise ValueError('Wrong comparison sample count')
        save_rows(OUT/f'predictions_{dataset}.csv',rows);datasets[dataset]=rows
        human=[]
        for r in rows:
            source=raw[KEY(r)];actual=float(r['target_pts'])
            if actual!=float(source['PTS']):raise ValueError('Actual source mismatch')
            h=dict(球員=r['player_id'],日期=r['GAMEDATE'],對戰=source['MATCHUP'],實際上場分鐘=float(source['MIN']),實際得分=actual,五場均值=float(r['pred_baseline']))
            for v in MODELS:
                if v.startswith('residual'):h[NAMES[v]+'預估增減']=r[v+'_delta']
                h[NAMES[v]]=r['pred_'+v];h[NAMES[v]+'絕對誤差']=abs(actual-r['pred_'+v])
            human.append(h)
        save_rows(OUT/f'四模型逐場比較_{dataset}.csv',human)
        s,a=summarize(dataset,rows);scores+=s;alerts+=a
        for first,second in PAIRS:
            for group in ['all','30+']:intervals.append(dict(dataset=dataset,first=first,second=second,group=group,**paired_comparison(rows,first,second,group)))
    save_rows(OUT/'scores.csv',scores);save_rows(OUT/'alert_metrics.csv',alerts);dump(OUT/'paired_intervals.json',intervals);make_report(scores,intervals)
    if any(sha(ROOT/p)!=h for p,h in before.items()):raise ValueError('Frozen sources changed')
    dump(OUT/'validation.json',dict(new_fits=0,new_offset_test_predictions=2*6979,reused_predictions_unchanged=True,
        original_test_usage_unchanged=True,production_promoted=False,source_sha256=before,
        code_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
        output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps([r for r in scores if r['dataset']=='Test' and r['group'] in ['all','30+']],ensure_ascii=False))


def make_report(scores,intervals):
    lines=['# 2025–26：五場均值增減與上季特徵模型比較','',
        '使用相同樣本、相同原固定參數與訓練cohort。CV9023、二月3462、三月至季末6979。兩種增減保留未加權與2倍，最終分數=max(0,五場均值＋預估增減)。本輪0fit、不調參，只補兩個凍結增減模型的test推論。','',
        '|期間|模型|整體MAE|RMSE|30+ MAE|<30 MAE|','|---|---|---:|---:|---:|---:|']
    for dataset in ['CV_pooled','February','Test']:
        for v in ALL:
            get=lambda group:next(r for r in scores if r['dataset']==dataset and r['variant']==v and r['group']==group)
            a=get('all');values=[a['mae'],a['rmse'],get('30+')['mae'],get('under30')['mae']]
            lines.append('|'+dataset+'|'+NAMES[v]+'|'+'|'.join(f'{x:.4f}' for x in values)+'|')
    lines+=['','本次結論：原九項＋上季表現的整體MAE在CV、二月及三月測試均最低。高分方面，增減2倍在三月至季末的點估計最低，但比新組合只低約0.060分；該差異的95%日期區間含0，沒有明確勝負。',
        '新組合比增減2倍的整體MAE低約0.074分，且CV／二月的30+誤差也較低。建議維持新兩方案作研究候選：九項上季觀察整體，組合上季2倍兼顧高分；增減模型沒有顯示足以替換它們的明確優勢。此建議不構成依已用測試調參或更換線上模型。','',
        '|三月後比較（新−舊）|分組|MAE差|95%日期區間|','|---|---|---:|---|']
    for r in intervals:
        if r['dataset']=='Test':lines.append(f"|{NAMES[r['first']]} → {NAMES[r['second']]}|{r['group']}|{r['delta_mae']:.4f}|[{r['lower_95']:.4f}, {r['upper_95']:.4f}]|")
    lines+=['','增減模型使用原1＋3＋同區20欄，新兩種使用14／25欄。因此比較的是完整方案，不能把差異全歸因於是否預測增減或單一上季特徵。',
        'CV／二月與三月test均已使用，屬追加研究；三月模型固定訓練至2026-01-31，CV各輪分別更早截止。先發為賽後快照，仍屬日期隔離的假設性回測。不是新獨立最終測試，也不依此追加調參或換線上模型。',
        '四模型逐場CSV包含原五場均值、兩種預估增減、最終得分、兩個新模型預測與各自絕對誤差。完整分箱／有無上季資料、30分警示與AP另存CSV。',
        'paired_intervals.json固定四個跨方案配對，每期all/30+共24個日期群組2000次區間，未作多重比較校正；負變化代表新prior方案誤差較低。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
