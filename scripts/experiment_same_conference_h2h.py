"""Eight fixed fits testing same-conference H2H memory on two existing bases."""
import argparse
import json
import platform
import importlib.metadata
from collections import defaultdict,Counter
from datetime import datetime,timezone
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits

from scripts.build_features_baseline import read_csv,sha
from scripts.compare_models import metrics,validate_development_rows
from scripts.experiment_recent_changes import ROOT,BASE,development_history,group_scores
from scripts.recent_change_features import NEW_FEATURES
from scripts.opponent_context import CONTEXT_FEATURES
from scripts.same_conference_h2h import H2H_FEATURES,h2h_features,h2h_matrix,SNAPSHOT
from scripts.experiment_two_stage import save_rows,control_predictions
from scripts.time_validation_tuning import build_folds,FitBudget,stratified_interval,dump,FOLDS
from nba_dashboard.data import expected_hash

VARIANTS={'base_h2h13':BASE+H2H_FEATURES,'case13_h2h20':BASE+NEW_FEATURES+CONTEXT_FEATURES+H2H_FEATURES}
ALL_VARIANTS=['common9','base_h2h13','case13','case13_h2h20']
PAIRS=[('common9','base_h2h13'),('case13','case13_h2h20'),('base_h2h13','case13_h2h20')]


def augment_rows(rows,history):
    players=defaultdict(list)
    for r in history: players[r['player_id']].append(r)
    return [dict(r,**h2h_features(players[r['player_id']],r)) for r in rows]


def coverage(period,rows):
    counts=Counter(r['h2h_status'] for r in rows)
    histogram=Counter(int(r['same_conf_h2h_count']) for r in rows if int(r['same_conference'])==1)
    same=counts['available']+counts['no_prior_meeting']
    return dict(period=period,n=len(rows),same_conference_n=same,with_prior_meeting_n=counts['available'],
                no_prior_meeting_n=counts['no_prior_meeting'],cross_conference_n=counts['cross_conference'],
                coverage_all=counts['available']/len(rows),coverage_within_same=counts['available']/same if same else None,
                same_conference_count_histogram=json.dumps(dict(sorted(histogram.items()))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/week2/same_conference_h2h_v6')
    out=parser.parse_args().out
    if out.exists(): raise ValueError('Existing output cannot be overwritten')
    prior=ROOT/'outputs/week2/opponent_context_v4'; old=ROOT/'outputs/week2/time_validation_tuning'
    raw=ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv'
    inputs=[raw,SNAPSHOT,old/'common_parameters.json',prior/'features/train.csv',prior/'features/validation.csv',
            prior/'predictions_CV_pooled.csv',prior/'predictions_February.csv',prior/'validation.json']
    protected=[p for folder in [old,ROOT/'outputs/week2/recent_change_v2',ROOT/'outputs/week2/two_stage_v3',prior,
                 ROOT/'outputs/week2/optimization_combinations_v5',ROOT/'outputs/week2/final_test'] for p in folder.rglob('*') if p.is_file()]
    protected += [ROOT/'streamlit_app.py',ROOT/'nba_dashboard/data.py',ROOT/'nba_dashboard/views.py',ROOT/'requirements.txt',ROOT/'requirements-modeling.txt']
    before={str(p.relative_to(ROOT)):sha(p) for p in set(inputs+protected)}
    prior_meta=json.loads(inputs[-1].read_text(encoding='utf-8'))
    for p in inputs[3:7]:
        if sha(p)!=expected_hash(prior_meta['output_sha256'],p.relative_to(prior)): raise ValueError('Changed prior artifact')
    if sha(raw)!=expected_hash(prior_meta['source_sha256'],raw.relative_to(ROOT)): raise ValueError('Changed raw snapshot')
    history=development_history(read_csv(raw))
    train,val=[augment_rows(read_csv(p),history) for p in inputs[3:5]]
    validate_development_rows(train,val)
    for rows in [train,val]: rows.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    if (len(train),len(val))!=(13221,3462): raise ValueError('Cohort changed; no H2H cold-start exclusion allowed')
    folds=build_folds(train)
    if [(len(tr),len(va)) for _,tr,va in folds]!=[(4198,4104),(8302,2422),(10724,2497)]: raise ValueError('Wrong folds')
    params=json.loads(inputs[2].read_text(encoding='utf-8'))['selected']['xgboost']['params']
    cv_controls=read_csv(inputs[5]); feb_controls=read_csv(inputs[6])
    out.mkdir(parents=True); (out/'features').mkdir(); (out/'models').mkdir()
    dump(out/'experiment_config.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),params=params,
         variants=VARIANTS,fit_cap=8,folds=FOLDS,common_train_n=13221,cv_n=9023,february_n=3462,
         h2h_scope='target same conference; same player vs opponent, this season, strictly prior date, MIN>0',
         mean_window='all prior current-season meetings, no fixed schedule count cap',transfer_history_preserved=True,
         missing_score_policy='native XGBoost NaN only in two H2H point columns; count=0; never fill with zero points',
         additional_exclusions=0,pairs=PAIRS,bootstrap_iterations=2000,seed=42,intervals_multiplicity_adjusted=False,
         test_scoring_allowed=False,auto_promotion=False,development_reused=True,exploratory_after_test_observed=True,
         conference_snapshot=str(SNAPSHOT.relative_to(ROOT))))
    save_rows(out/'features/train.csv',train); save_rows(out/'features/validation.csv',val)
    save_rows(out/'feature_dictionary.csv',[
        dict(feature=H2H_FEATURES[0],definition='本場球隊與對手同conference為1，否則0'),
        dict(feature=H2H_FEATURES[1],definition='本季此前球員對同一對手的正分鐘次數；本場跨區固定0'),
        dict(feature=H2H_FEATURES[2],definition='最近一次得分；沒有紀錄或跨區留空，不補0'),
        dict(feature=H2H_FEATURES[3],definition='此前所有本季對陣平均得分；無固定三或四場上限，無紀錄留空')])
    budget=FitBudget(8); fit_log=[]; predictions={}; reloaded=True
    with threadpool_limits(limits=2):
        for period,tr,va in folds+[('February',train,val)]:
            controls=[r for r in cv_controls if r['period']==period] if period!='February' else feb_controls
            common9=control_predictions(va,controls,'pred_common9'); case13=control_predictions(va,controls,'pred_opponent16')
            frozen9=control_predictions(va,controls,'pred_frozen9')
            rows=[{k:r[k] for k in ['player_id','game_id','GAMEDATE','target_pts',*H2H_FEATURES,
                                    'h2h_status','h2h_history_game_ids','h2h_last_history_date']}|
                dict(period=period,pred_baseline=float(r['pts_mean_5']),pred_common9=float(common9[i]),
                     pred_case13=float(case13[i]),pred_frozen9=float(frozen9[i])) for i,r in enumerate(va)]
            without_labels=[{k:v for k,v in r.items() if not k.startswith('target_')} for r in va]
            for variant,columns in VARIANTS.items():
                model=XGBRegressor(**params); budget.consume()
                model.fit(h2h_matrix(tr,columns),[float(r['target_pts']) for r in tr])
                pred=model.predict(h2h_matrix(va,columns)).astype(float)
                if not np.isfinite(pred).all() or not np.array_equal(pred,model.predict(h2h_matrix(without_labels,columns)).astype(float)):
                    raise ValueError('Invalid or outcome-dependent prediction')
                path=out/'models'/f'{period}_{variant}.joblib'
                joblib.dump(dict(model=model,columns=columns,params=params,n_train=len(tr),
                    trained_through=max(r['GAMEDATE'] for r in tr),research_only=True,missing_scores_are_nan=True),path)
                loaded=joblib.load(path)
                reloaded &= np.array_equal(pred,loaded['model'].predict(h2h_matrix(without_labels,loaded['columns'])).astype(float))
                for r,p in zip(rows,pred): r['pred_'+variant]=float(p)
                fit_log.append(dict(period=period,variant=variant,fit_number=budget.count,n_features=len(columns),
                    n_train=len(tr),n_validation=len(va),train_end=max(r['GAMEDATE'] for r in tr),validation_start=min(r['GAMEDATE'] for r in va)))
            predictions[period]=rows
    cv=sum([predictions[name] for name,_,_ in folds],[]); predictions['CV_pooled']=cv
    if len(cv)!=9023 or len({(r['player_id'],r['game_id']) for r in cv})!=9023: raise ValueError('Wrong pooled cohort')
    scores=[]; groups=[]; references=[]; intervals=[]; coverage_rows=[coverage('train',train)]
    for period,rows in predictions.items():
        y=[float(r['target_pts']) for r in rows]; baseline=[r['pred_baseline'] for r in rows]
        coverage_rows.append(coverage(period,rows))
        for variant in ALL_VARIANTS:
            scores.extend(dict(period=period,variant=variant,**s) for s in group_scores(y,[r['pred_'+variant] for r in rows],baseline))
            for status in ['available','no_prior_meeting','cross_conference']:
                part=[r for r in rows if r['h2h_status']==status]
                score=metrics([r['target_pts'] for r in part],[r['pred_'+variant] for r in part]) if part else dict(n=0,mae=None,rmse=None)
                groups.append(dict(period=period,variant=variant,h2h_status=status,**score))
        for variant in ['baseline','frozen9']:
            references.extend(dict(period=period,variant=variant,**s) for s in group_scores(y,[r['pred_'+variant] for r in rows],baseline))
        if period in ['CV_pooled','February']:
            save_rows(out/f'predictions_{period}.csv',rows)
            for first,second in PAIRS:
                diff=stratified_interval(y,[r['pred_'+first] for r in rows],[r['pred_'+second] for r in rows],
                    [r['GAMEDATE'] for r in rows],[r['period'] for r in rows])
                intervals.append(dict(period=period,first=first,second=second,**diff))
    for name,rows in [('scores',scores),('reference_scores',references),('h2h_group_scores',groups),
                      ('coverage',coverage_rows),('fit_log',fit_log)]: save_rows(out/f'{name}.csv',rows)
    dump(out/'paired_intervals.json',intervals)
    make_report(out,scores,groups,coverage_rows,intervals)
    if budget.count!=8 or not reloaded or any(sha(ROOT/p)!=h for p,h in before.items()): raise ValueError('Fit/reload/protected source check failed')
    code=[Path(__file__),ROOT/'scripts/same_conference_h2h.py',ROOT/'scripts/experiment_recent_changes.py',
          ROOT/'scripts/time_validation_tuning.py',ROOT/'scripts/experiment_two_stage.py',ROOT/'scripts/opponent_context.py']
    dump(out/'validation.json',dict(fits=8,train_n=13221,cv_n=9023,february_n=3462,additional_exclusions=0,
         exact_controls=True,missing_scores_preserved=True,model_reload_equal=bool(reloaded),test_metrics_computed=False,
         production_promoted=False,source_sha256=before,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},
         environment=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in
                          ['numpy','scikit-learn','xgboost','joblib','matplotlib']}),
         output_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(output=str(out),fits=8,intervals=intervals),ensure_ascii=False))


def make_report(out,scores,groups,coverage_rows,intervals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=dict(common9='共同九項',base_h2h13='九項＋同區對陣13項',case13='1＋3原16項',case13_h2h20='1＋3＋同區對陣20項')
    lines=['# 同區先前對陣得分特徵','',
        '新增本場是否同conference、此前對陣次數、最近一次得分、此前本季對陣平均。轉隊延續個人紀錄，跨季重置，僅使用嚴格前日正分鐘出賽。',
        '只在本場同區啟用得分記憶；跨區或首次交手的兩個得分欄留空，由XGBoost原生缺失處理，絕不填0分。正分鐘真的得0分仍保留。',
        '按實際紀錄，不硬設每季兩場／三場／四場上限；固定2025–26東西區對照見 docs/week1/conferences_2025_26.json，核對日不是歷史發布時間。',
        '分區官方來源：[NBA About](https://www.nba.com/news/about)、[2025–26 standings](https://www.nba.com/standings?Season=2025-26)。',
        '同一13221筆訓練與9023／3462筆驗證，沒有因首次交手額外排除；固定8fit不調參，不讀三月後結果或替換線上模型。','',
        '|期間|版本|樣本|MAE|RMSE|','|---|---|---:|---:|---:|']
    for period in ['F1','F2','F3','CV_pooled','February']:
        for variant in ALL_VARIANTS:
            r=next(r for r in scores if r['period']==period and r['variant']==variant and r['group']=='all')
            lines.append(f"|{period}|{labels[variant]}|{r['n']}|{r['mae']:.4f}|{r['rmse']:.4f}|")
    lines+=['','## 有多少對陣紀錄？','', '|期間|全部樣本|同區|同區有對陣|同區覆蓋率|','|---|---:|---:|---:|---:|']
    for r in coverage_rows:
        if r['period'] in ['train','CV_pooled','February']:
            lines.append(f"|{r['period']}|{r['n']}|{r['same_conference_n']}|{r['with_prior_meeting_n']}|{r['coverage_within_same']:.1%}|")
    lines+=['','## 高得分場次','', '|期間|得分組|版本|樣本|MAE|RMSE|','|---|---|---|---:|---:|---:|']
    for r in scores:
        if r['period'] in ['CV_pooled','February'] and r['group']!='all':
            lines.append(f"|{r['period']}|{r['group']}|{labels[r['variant']]}|{r['n']}|{r['mae']:.4f}|{r['rmse']:.4f}|")
    lines+=['','## 成對差異','', '後者減前者MAE，負數為改善。日期群組重抽2000次、CV輪內分層；未作多重比較修正。']
    for r in intervals:
        lines.append(f"- {r['period']}／{labels[r['second']]}減{labels[r['first']]}：{r['delta_mae']:.4f}，95%區間 [{r['lower_95']:.4f}, {r['upper_95']:.4f}]。")
    lines+=['','## 限制','',
        '每個球員對特定對手的樣本通常很少；上次高分可能來自當時的分鐘、命中率或陣容，不能直接解釋為對手造成，也不能直接把上次得分當預測。',
        '這是看過舊測試後、重用開發資料的探索；同區有紀錄、同區無紀錄、跨區三群的誤差見 h2h_group_scores.csv，不能只報有紀錄群而隱藏整體。',
        '新球季分區對照須另核對；本次不抓取新來源結果、不產生三月後預測。原九項正式模型與線上頁面保持不變。',
        '日期隔離的假設性回測，以出賽為前提；先發為賽後快照，對手背景不是正式Pace，正式採用仍需新資料驗證。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,period in zip(axes,['CV_pooled','February']):
        values=[next(r['mae'] for r in scores if r['period']==period and r['variant']==v and r['group']=='all') for v in ALL_VARIANTS]
        ax.bar(['Base9','Base+H2H13','1+3 (16)','1+3+H2H20'],values)
        for i,v in enumerate(values): ax.text(i,v,f'{v:.4f}',ha='center',va='bottom')
        ax.set(title=period,ylabel='Points MAE',ylim=(0,max(values)*1.15))
    fig.tight_layout(); fig.savefig(out/'mae_h2h_comparison.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    main()
