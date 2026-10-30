"""Fixed sixteen-fit tail-weight study: baseline9 first, combined20 second."""
import argparse
import json
import platform
import importlib.metadata
from datetime import datetime,timezone
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits

from scripts.build_features_baseline import read_csv,sha
from scripts.compare_models import validate_development_rows
from scripts.experiment_recent_changes import ROOT
from scripts.experiment_two_stage import save_rows,control_predictions
from scripts.high_score_weighting import VARIANTS,PAIRS,high_score_weights,input_matrix,point_group_scores,alert_metrics,paired_comparison
from scripts.time_validation_tuning import build_folds,FitBudget,dump,FOLDS
from nba_dashboard.data import expected_hash

ALL=['basic_unweighted','basic_w2','basic_w3','combined_unweighted','combined_w2','combined_w3']
NAMES={'basic_unweighted':'原九項／未加權','basic_w2':'原九項／高分2倍','basic_w3':'原九項／高分3倍',
       'combined_unweighted':'1＋3＋同區20項／未加權','combined_w2':'20項／高分2倍','combined_w3':'20項／高分3倍'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/week2/high_score_weighting_v7')
    out=parser.parse_args().out
    if out.exists(): raise ValueError('Refuse to overwrite existing experiment')
    prior=ROOT/'outputs/week2/same_conference_h2h_v6'; old=ROOT/'outputs/week2/time_validation_tuning'
    inputs=[prior/'features/train.csv',prior/'features/validation.csv',prior/'predictions_CV_pooled.csv',
            prior/'predictions_February.csv',prior/'validation.json',old/'common_parameters.json']
    protected=[p for folder in [old,ROOT/'outputs/week2/recent_change_v2',ROOT/'outputs/week2/two_stage_v3',
                 ROOT/'outputs/week2/opponent_context_v4',ROOT/'outputs/week2/optimization_combinations_v5',prior,
                 ROOT/'outputs/week2/final_test'] for p in folder.rglob('*') if p.is_file()]
    protected += [ROOT/'streamlit_app.py',ROOT/'nba_dashboard/data.py',ROOT/'nba_dashboard/views.py',
                  ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv',
                  ROOT/'requirements.txt',ROOT/'requirements-modeling.txt']
    before={str(p.relative_to(ROOT)):sha(p) for p in set(inputs+protected)}
    prior_meta=json.loads(inputs[4].read_text(encoding='utf-8'))
    for p in inputs[:4]:
        if sha(p)!=expected_hash(prior_meta['output_sha256'],p.relative_to(prior)): raise ValueError('Changed common data/control')
    train,val=[read_csv(p) for p in inputs[:2]]; validate_development_rows(train,val)
    for rows in [train,val]: rows.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    if (len(train),len(val))!=(13221,3462): raise ValueError('Changed common cohort')
    folds=build_folds(train)
    if [(len(tr),len(va)) for _,tr,va in folds]!=[(4198,4104),(8302,2422),(10724,2497)]: raise ValueError('Changed folds')
    params=json.loads(inputs[5].read_text(encoding='utf-8'))['selected']['xgboost']['params']
    cv_controls=read_csv(inputs[2]); feb_controls=read_csv(inputs[3])
    out.mkdir(parents=True); (out/'models').mkdir()
    dump(out/'experiment_config.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        variants=VARIANTS,params=params,fit_cap=16,folds=FOLDS,common_train_n=13221,cv_n=9023,february_n=3462,
        train_threshold=30,raw_weight='multiplier if train target_pts>=30 else 1',
        weight_normalization='divide raw weights by training-only mean',evaluation_weighted=False,
        alert_threshold=30,ap_uses_regression_ranking_not_probabilities=True,pairs=PAIRS,
        paired_groups=['all','30+'],bootstrap_iterations=2000,seed=42,multiplicity_adjusted=False,
        test_scoring_allowed=False,auto_promotion=False,development_reused=True,exploratory_after_test_observed=True,
        primary_focus='high-score error plus explicit under30/overall cost and false alerts; no user-approved tradeoff cutoff'))
    budget=FitBudget(16); predictions={}; fit_log=[]; reload_equal=True
    with threadpool_limits(limits=2):
        for period,tr,va in folds+[('February',train,val)]:
            controls=[r for r in cv_controls if r['period']==period] if period!='February' else feb_controls
            basic=control_predictions(va,controls,'pred_common9'); combined=control_predictions(va,controls,'pred_case13_h2h20')
            frozen9=control_predictions(va,controls,'pred_frozen9')
            rows=[{k:r[k] for k in ['player_id','game_id','GAMEDATE','target_pts']}|
                  dict(period=period,pred_baseline=float(r['pts_mean_5']),pred_frozen9=float(frozen9[i]),
                       pred_basic_unweighted=float(basic[i]),pred_combined_unweighted=float(combined[i])) for i,r in enumerate(va)]
            labels=np.asarray([float(r['target_pts']) for r in tr])
            without_labels=[{k:v for k,v in r.items() if not k.startswith('target_')} for r in va]
            fake_labels=[dict(r,target_pts=99999,target_minutes=99999) for r in va]
            for variant,spec in VARIANTS.items():
                weights,normalizer=high_score_weights(labels,spec['multiplier'])
                model=XGBRegressor(**params); budget.consume()
                model.fit(input_matrix(tr,variant),labels,sample_weight=weights)
                pred=model.predict(input_matrix(va,variant)).astype(float)
                if (not np.isfinite(pred).all() or not np.array_equal(pred,model.predict(input_matrix(without_labels,variant)).astype(float))
                    or not np.array_equal(pred,model.predict(input_matrix(fake_labels,variant)).astype(float))):
                    raise ValueError('Invalid or label-dependent prediction')
                path=out/'models'/f'{period}_{variant}.joblib'
                joblib.dump(dict(model=model,columns=spec['columns'],params=params,variant=variant,
                    weight_multiplier=spec['multiplier'],raw_weight_normalizer=normalizer,n_train=len(tr),
                    trained_through=max(r['GAMEDATE'] for r in tr),research_only=True),path)
                loaded=joblib.load(path)
                reload_equal &= np.array_equal(pred,loaded['model'].predict(input_matrix(without_labels,variant)).astype(float))
                for r,p in zip(rows,pred): r['pred_'+variant]=float(p)
                high_n=int((labels>=30).sum())
                fit_log.append(dict(period=period,variant=variant,fit_number=budget.count,n_features=len(spec['columns']),
                    n_train=len(tr),train_high_n=high_n,multiplier=spec['multiplier'],raw_weight_normalizer=normalizer,
                    low_effective_weight=1/normalizer,high_effective_weight=spec['multiplier']/normalizer,
                    effective_weight_mean=float(weights.mean()),effective_high_share=float(weights[labels>=30].sum()/weights.sum()),
                    n_validation=len(va),train_end=max(r['GAMEDATE'] for r in tr),validation_start=min(r['GAMEDATE'] for r in va)))
            predictions[period]=rows
    cv=sum([predictions[name] for name,_,_ in folds],[]); predictions['CV_pooled']=cv
    if len(cv)!=9023 or len({(r['player_id'],r['game_id']) for r in cv})!=9023: raise ValueError('Wrong pooled cohort')
    scores=[]; references=[]; alerts=[]; intervals=[]
    for period,rows in predictions.items():
        actual=[float(r['target_pts']) for r in rows]
        for variant in ALL:
            predicted=[r['pred_'+variant] for r in rows]
            scores.extend(dict(period=period,variant=variant,**s) for s in point_group_scores(actual,predicted))
            alerts.append(dict(period=period,variant=variant,**alert_metrics(actual,predicted)))
        for variant in ['baseline','frozen9']:
            references.extend(dict(period=period,variant=variant,**s) for s in point_group_scores(actual,[r['pred_'+variant] for r in rows]))
        if period in ['CV_pooled','February']:
            save_rows(out/f'predictions_{period}.csv',rows)
            for first,second in PAIRS:
                for group in ['all','30+']:
                    intervals.append(dict(period=period,first=first,second=second,group=group,**paired_comparison(rows,first,second,group)))
    for name,rows in [('scores',scores),('reference_scores',references),('alert_metrics',alerts),('fit_log',fit_log)]: save_rows(out/f'{name}.csv',rows)
    dump(out/'paired_intervals.json',intervals)
    make_report(out,scores,alerts,intervals)
    if budget.count!=16 or not reload_equal or any(sha(ROOT/p)!=h for p,h in before.items()): raise ValueError('Budget/reload/protected source check failed')
    code=[Path(__file__),ROOT/'scripts/high_score_weighting.py',ROOT/'scripts/same_conference_h2h.py',
          ROOT/'scripts/experiment_recent_changes.py',ROOT/'scripts/time_validation_tuning.py',ROOT/'scripts/compare_models.py']
    dump(out/'validation.json',dict(fits=16,common_train_n=13221,cv_n=9023,february_n=3462,
        training_only_weights=True,evaluation_unweighted=True,controls_exact=True,model_reload_equal=bool(reload_equal),
        inference_without_or_fake_labels_equal=True,test_metrics_computed=False,production_promoted=False,
        source_sha256=before,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},
        environment=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in
                         ['numpy','scikit-learn','xgboost','joblib','matplotlib']}),
        output_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(output=str(out),fits=16,pooled_and_february_scores=[s for s in scores if s['period'] in ['CV_pooled','February'] and s['group'] in ['all','30+']]),ensure_ascii=False))


def make_report(out,scores,alerts,intervals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines=['# 高分樣本加權：先單獨，再加入1＋3與同區','',
        '第一步九項直接得分模型，只改訓練權重；第二步使用1＋3＋同區對陣20項，同樣只改權重。高分固定PTS>=30，倍率2／3，其餘1，除以當輪訓練raw權重平均。',
        '所有指標一律不加權，沒有重抽樣、改分數、加入實際高分旗標或事後調高預測。評估資料不用來設定訓練權重。',
        '四個加權版本×四個期間共16fit，參數／cohort固定，未加權控制讀先前凍結預測；共同訓練13221，CV9023／二月3462，原早期60筆對手不足五場的排除全體一致。','',
        '|期間|模型|整體MAE|整體RMSE|30+ MAE|30+平均低估|<30 MAE|','|---|---|---:|---:|---:|---:|---:|']
    for period in ['CV_pooled','February']:
        for variant in ALL:
            get=lambda group:next(s for s in scores if s['period']==period and s['variant']==variant and s['group']==group)
            a,h,low=[get(g) for g in ['all','30+','under30']]
            lines.append(f"|{period}|{NAMES[variant]}|{a['mae']:.4f}|{a['rmse']:.4f}|{h['mae']:.4f}|{h['mean_residual']:.4f}|{low['mae']:.4f}|")
    lines+=['','## 是否真正抓到高分？','',
        '固定預測>=30分才發高分警示。Precision＝警示中真的高分比例，Recall＝實際高分被抓到的比例。沒有警示時Precision留白，Recall仍依真值算。',
        'AP用預測分數排序，不是高分機率；應與高分盛行率一起看，不能把回歸分數直接當機率。','',
        '|期間|模型|實際高分|警示|抓到TP|誤報FP|Precision|Recall|AP|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in alerts:
        if r['period'] in ['CV_pooled','February']:
            precision='—' if r['precision'] is None else f"{r['precision']:.1%}"
            lines.append(f"|{r['period']}|{NAMES[r['variant']]}|{r['actual_high_n']}|{r['predicted_high_n']}|{r['tp']}|{r['fp']}|{precision}|{r['recall']:.1%}|{r['average_precision']:.4f}|")
    lines+=['','## 各輪穩定性','', '|輪次|模型|整體MAE|30+ MAE|<30 MAE|','|---|---|---:|---:|---:|']
    for period in ['F1','F2','F3']:
        for variant in ALL:
            vals=[next(s['mae'] for s in scores if s['period']==period and s['variant']==variant and s['group']==g) for g in ['all','30+','under30']]
            lines.append(f"|{period}|{NAMES[variant]}|{vals[0]:.4f}|{vals[1]:.4f}|{vals[2]:.4f}|")
    lines+=['','## 固定成對比較','',
        '後者減前者MAE，負值為改善。整體與實際30+分別報告；日期群組2000次、CV輪內分層。區間不做多重比較修正。']
    for r in intervals:
        if r['status']=='estimated':
            lines.append(f"- {r['period']}／{r['group']}／{NAMES[r['second']]}減{NAMES[r['first']]}：{r['delta_mae']:.4f}，95%區間 [{r['lower_95']:.4f}, {r['upper_95']:.4f}]。")
        else: lines.append(f"- {r['period']}／{r['group']}：某層日期不足兩天，不估計區間。")
    lines+=['','## 限制與判讀','',
        '高分加權改變訓練目標，可能把整體預測往上拉。高分MAE下降必須與<30誤差、高估比例、誤報和召回一起解讀；不能只報高分子集。',
        '高分與各得分區間為賽後診斷，不可用真值決定要啟用哪個模型。警示只用預測值，閾值固定30，沒有用結果追加調整。',
        '使用者未指定可接受的整體退步幅度，因此本報告呈現取捨，不宣告可部署、不自動更換線上模型。',
        '這是看過舊測試後並重用開發資料的探索。三月至四月不產生新預測、不重評測試；正式採用須新資料預定協議驗收。',
        '仍屬日期隔離的假設性回測，以球員出賽為前提；歷史先發為賽後快照，對手背景不是正式Pace，同區得分缺失仍保留NaN。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for row,period in enumerate(['CV_pooled','February']):
        for col,group in enumerate(['all','30+']):
            ax=axes[row,col]
            values=[next(s['mae'] for s in scores if s['period']==period and s['variant']==v and s['group']==group) for v in ALL]
            ax.bar(['9 x1','9 x2','9 x3','20 x1','20 x2','20 x3'],values)
            for i,v in enumerate(values): ax.text(i,v,f'{v:.3f}',ha='center',va='bottom')
            ax.set(title=f'{period} / {group}',ylabel='Unweighted points MAE',ylim=(0,max(values)*1.13))
    fig.tight_layout(); fig.savefig(out/'high_score_tradeoff.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    main()
