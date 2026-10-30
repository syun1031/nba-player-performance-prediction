"""Four fixed combinations, 28 fits, common development cohort only."""
import argparse
import json
import platform
import importlib.metadata
from datetime import datetime,timezone
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits

from scripts.build_features_baseline import read_csv,sha
from scripts.compare_models import metrics,validate_development_rows
from scripts.experiment_recent_changes import ROOT,matrix,development_history,group_scores
from scripts.experiment_two_stage import save_rows,control_predictions
from scripts.two_stage_model import attach_minute_targets
from scripts.combination_models import CASES,CombinationTwoStage,rank_cases
from scripts.time_validation_tuning import build_folds,FitBudget,stratified_interval,dump,FOLDS
from nba_dashboard.data import expected_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/week2/optimization_combinations_v5')
    out=parser.parse_args().out
    if out.exists():
        raise ValueError('Existing output cannot be overwritten')
    context=ROOT/'outputs/week2/opponent_context_v4'; old=ROOT/'outputs/week2/time_validation_tuning'
    raw=ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv'
    inputs=[raw,old/'common_parameters.json',context/'features/train.csv',context/'features/validation.csv',
            context/'predictions_CV_pooled.csv',context/'predictions_February.csv',context/'validation.json']
    protected=[p for folder in [old,ROOT/'outputs/week2/recent_change_v2',ROOT/'outputs/week2/two_stage_v3',
                  context,ROOT/'outputs/week2/final_test'] for p in folder.rglob('*') if p.is_file()]
    protected += [ROOT/'streamlit_app.py',ROOT/'nba_dashboard/data.py',ROOT/'nba_dashboard/views.py',
                  ROOT/'requirements.txt',ROOT/'requirements-modeling.txt',ROOT/'scripts/two_stage_model.py',
                  ROOT/'scripts/opponent_context.py']
    before={str(p.relative_to(ROOT)):sha(p) for p in set(inputs+protected)}
    prior=json.loads(inputs[-1].read_text(encoding='utf-8'))
    for p in inputs[2:6]:
        if sha(p)!=expected_hash(prior['output_sha256'],p.relative_to(context)):
            raise ValueError('Common feature/control artifact changed')
    if sha(raw)!=expected_hash(prior['source_sha256'],raw.relative_to(ROOT)):
        raise ValueError('Raw label source changed')
    params=json.loads(inputs[1].read_text(encoding='utf-8'))['selected']['xgboost']['params']
    history=development_history(read_csv(raw))
    train,val=[attach_minute_targets(read_csv(p),history) for p in inputs[2:4]]
    validate_development_rows(train,val)
    for rows in [train,val]: rows.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    if (len(train),len(val))!=(13221,3462) or any(int(r['opponent_history_count'])<5 for r in train+val):
        raise ValueError('Common cohort/eligibility changed')
    folds=build_folds(train)
    if [(len(tr),len(va)) for _,tr,va in folds]!=[(4198,4104),(8302,2422),(10724,2497)]:
        raise ValueError('Fixed fold sizes changed')
    cv_controls=read_csv(inputs[4]); feb_controls=read_csv(inputs[5])
    out.mkdir(parents=True); (out/'models').mkdir(); (out/'features').mkdir()
    dump(out/'experiment_config.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
         cases=CASES,params=params,fit_cap=28,folds=FOLDS,common_train_n=13221,excluded_early_train_n=60,
         eval_cv_n=9023,eval_february_n=3462,primary_ranking='CV pooled MAE, RMSE, fewer features, fewer fits, case id',
         rate_sample_weight='training MIN / training mean MIN',nonnegative_components=True,overtime_cap=None,
         pairwise_comparisons=list(combinations(CASES,2)),bootstrap_iterations=2000,seed=42,
         intervals_multiplicity_adjusted=False,development_reused=True,exploratory_after_test_observed=True,
         test_scoring_allowed=False,auto_promotion=False,true_pace_available=False))
    save_rows(out/'features/train.csv',train); save_rows(out/'features/validation.csv',val)
    budget=FitBudget(28); fit_log=[]; predictions={}; reload_equal=True
    with threadpool_limits(limits=2):
        for period,tr,va in folds+[('February',train,val)]:
            controls=[r for r in cv_controls if r['period']==period] if period!='February' else feb_controls
            expected13=control_predictions(va,controls,'pred_opponent16')
            frozen9=control_predictions(va,controls,'pred_frozen9')
            rows=[{k:r[k] for k in ['player_id','game_id','GAMEDATE','target_pts','target_minutes','min_mean_5']}|
                  dict(period=period,pred_baseline=float(r['pts_mean_5']),pred_frozen9=float(frozen9[i])) for i,r in enumerate(va)]
            without_labels=[{k:v for k,v in r.items() if not k.startswith('target_')} for r in va]
            for case,spec in CASES.items():
                start_count=budget.count; columns=spec['columns']
                if spec['mode']=='two_stage':
                    model=CombinationTwoStage(params,case).fit(tr,budget)
                    result=model.predict_components(va); pred=result['points']
                    no_labels=model.predict_components(without_labels)
                    if any(not np.array_equal(result[k],no_labels[k]) for k in result):
                        raise ValueError('Outcome labels influence two-stage prediction')
                    for i,r in enumerate(rows):
                        for component in ['minutes','rate','raw_minutes','raw_rate']:
                            r[f'{case}_{component}']=float(result[component][i])
                    bundle=dict(model=model,columns=columns,case=case,mode=spec['mode'],params=params,
                                n_train=len(tr),trained_through=model.trained_through,research_only=True)
                else:
                    model=XGBRegressor(**params); budget.consume()
                    model.fit(matrix(tr,columns),[float(r['target_pts']) for r in tr])
                    pred=model.predict(matrix(va,columns)).astype(float)
                    if not np.array_equal(pred,model.predict(matrix(without_labels,columns)).astype(float)):
                        raise ValueError('Outcome labels influence direct prediction')
                    bundle=dict(model=model,columns=columns,case=case,mode=spec['mode'],params=params,
                                n_train=len(tr),trained_through=max(r['GAMEDATE'] for r in tr),research_only=True)
                if not np.isfinite(pred).all() or budget.count-start_count!=spec['fits']:
                    raise ValueError('Prediction/budget mismatch')
                if case=='case13' and not np.array_equal(pred,expected13):
                    raise ValueError('Case13 differs from identical prior opponent16 model')
                path=out/'models'/f'{period}_{case}.joblib'; joblib.dump(bundle,path); loaded=joblib.load(path)
                if spec['mode']=='two_stage':
                    again=loaded['model'].predict_components(without_labels)
                    reload_equal &= all(np.array_equal(result[k],again[k]) for k in result)
                else:
                    reload_equal &= np.array_equal(pred,loaded['model'].predict(matrix(without_labels,loaded['columns'])).astype(float))
                for r,p in zip(rows,pred): r['pred_'+case]=float(p)
                stages=['minutes','rate'] if spec['mode']=='two_stage' else ['points']
                for stage in stages:
                    fit_log.append(dict(period=period,case=case,stage=stage,fit_number=len(fit_log)+1,
                        n_features=len(columns),n_train=len(tr),n_validation=len(va),
                        train_end=max(r['GAMEDATE'] for r in tr),validation_start=min(r['GAMEDATE'] for r in va)))
            predictions[period]=rows
    cv=sum([predictions[name] for name,_,_ in folds],[]); predictions['CV_pooled']=cv
    if len(cv)!=9023 or len({(r['player_id'],r['game_id']) for r in cv})!=9023:
        raise ValueError('Pooled validation keys differ')
    scores=[]; references=[]; minute_scores=[]; intervals=[]; rankings={}
    for period,rows in predictions.items():
        y=[r['target_pts'] for r in rows]; base=[r['pred_baseline'] for r in rows]
        for case,spec in CASES.items():
            scores.extend(dict(period=period,variant=case,**s) for s in group_scores(y,[r['pred_'+case] for r in rows],base))
            if spec['mode']=='two_stage':
                minute_scores.append(dict(period=period,variant=case,**metrics([r['target_minutes'] for r in rows],[r[case+'_minutes'] for r in rows])))
        for reference in ['baseline','frozen9']:
            references.extend(dict(period=period,variant=reference,**s) for s in group_scores(y,[r['pred_'+reference] for r in rows],base))
        ranking=rank_cases([r for r in scores if r['period']==period and r['group']=='all'])
        rankings[period]=[dict(rank=i+1,**r) for i,r in enumerate(ranking)]
        if period in ['CV_pooled','February']:
            save_rows(out/f'predictions_{period}.csv',rows)
            for first,second in combinations(CASES,2):
                diff=stratified_interval(y,[r['pred_'+first] for r in rows],[r['pred_'+second] for r in rows],
                    [r['GAMEDATE'] for r in rows],[r['period'] for r in rows])
                intervals.append(dict(period=period,first=first,second=second,**diff))
    for name,rows in [('scores',scores),('reference_scores',references),('minute_scores',minute_scores),('fit_log',fit_log)]:
        save_rows(out/f'{name}.csv',rows)
    dump(out/'rankings.json',rankings); dump(out/'paired_intervals.json',intervals)
    make_report(out,scores,references,rankings,intervals)
    if budget.count!=28 or len(fit_log)!=28 or not reload_equal or any(sha(ROOT/p)!=h for p,h in before.items()):
        raise ValueError('Budget/reload/protected artifact check failed')
    code=[Path(__file__),ROOT/'scripts/combination_models.py',ROOT/'scripts/two_stage_model.py',
          ROOT/'scripts/time_validation_tuning.py',ROOT/'scripts/experiment_two_stage.py']
    dump(out/'validation.json',dict(fits=28,common_train_n=13221,cv_n=9023,february_n=3462,case13_exact=True,
        model_bundles=16,underlying_models=28,model_reload_equal=bool(reload_equal),prediction_without_labels_equal=True,
        all_cases_same_cohort=True,test_metrics_computed=False,production_promoted=False,
        source_sha256=before,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},
        environment=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in
                         ['numpy','scikit-learn','xgboost','joblib','matplotlib']}),
        output_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(output=str(out),fits=28,cv_ranking=rankings['CV_pooled'],february_ranking=rankings['February']),ensure_ascii=False))


def make_report(out,scores,references,rankings,intervals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines=['# 三項優化：四種組合比較','',
        '1＝五項近期變化特徵；2＝預估分鐘×按分鐘加權的預估得分效率；3＝對手近五場失分／雙方總分背景（不是正式Pace）。',
        '同一13221筆訓練、相同日期切分及驗證樣本；四組一起排除早期60筆對手不足五場的樣本。每輪7fit、四輪共28次，固定原參數、不追加搜尋。',
        '1＋2：14項兩階段；1＋3：16項直接得分；2＋3：11項兩階段（沒有1）；1＋2＋3：16項兩階段。',
        '主排名先看三輪逐筆合併MAE，再RMSE、較少特徵、較少fit與編號；二月另外排名，未拿三月至四月舊測試選組合。','',
        '|期間|排名|組合|MAE|RMSE|','|---|---:|---|---:|---:|']
    for period in ['CV_pooled','February','F1','F2','F3']:
        for r in rankings[period]:
            lines.append(f"|{period}|{r['rank']}|{CASES[r['variant']]['label']}|{r['mae']:.4f}|{r['rmse']:.4f}|")
    primary=rankings['CV_pooled'][0]; feb=rankings['February'][0]
    lines+=['',f"按預定主排名，三輪最佳為 **{CASES[primary['variant']]['label']}**，二月最佳為 **{CASES[feb['variant']]['label']}**。",
            '兩期最佳相同。' if primary['variant']==feb['variant'] else '兩期最佳不同，不能只挑較好月份宣稱同一組穩定勝出。',
            '', '## 高得分場次','', '|期間|實際得分組|樣本|組合|MAE|RMSE|','|---|---|---:|---|---:|---:|']
    for r in scores:
        if r['period'] in ['CV_pooled','February'] and r['group']!='all':
            lines.append(f"|{r['period']}|{r['group']}|{r['n']}|{CASES[r['variant']]['label']}|{r['mae']:.4f}|{r['rmse']:.4f}|")
    lines+=['','## 固定六組成對比較','',
            '差異＝後列組合減前列組合的MAE，負值代表後列較好。每期間2000次日期群組重抽，CV輪內分層。區間未做多重比較修正，不是挑贏家後才補做的單次檢定。']
    for r in intervals:
        lines.append(f"- {r['period']}：{CASES[r['second']]['label']} 減 {CASES[r['first']]['label']} = {r['delta_mae']:.4f}，95%區間 [{r['lower_95']:.4f}, {r['upper_95']:.4f}]。")
    lines+=['','## 參考與限制','', '|期間|參考|MAE|RMSE|','|---|---|---:|---:|']
    for r in references:
        if r['period'] in ['CV_pooled','February'] and r['group']=='all':
            label='五場均值' if r['variant']=='baseline' else '原凍結九項'
            lines.append(f"|{r['period']}|{label}|{r['mae']:.4f}|{r['rmse']:.4f}|")
    lines+=['','原凍結九項訓練13281筆，和本輪共同13221筆不同，只作目前線上版本參考。1＋3已逐筆重現上一輪相同樣本與設定的opponent16。',
            '本輪三組兩階段模型的分鐘評分另見 minute_scores.csv，實際分鐘只當訓練標籤／權重與評估，不進入預測輸入。',
            '這是看過舊測試後、重用開發資料並比較多組配置的探索排名，不代表未見資料上已證明更好。最終測試不重評，線上模型不自動更換。',
            '需同時檢查各輪、二月及20+／30+，平均誤差較低不保證爆分預測較準。正式採用前需新資料另定驗證。',
            '日期隔離的假設性回測，以出賽為前提；歷史先發為賽後快照，對手總分含延長賽，不是完整賽前重建或正式Pace。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,period in zip(axes,['CV_pooled','February']):
        values=[next(r['mae'] for r in scores if r['period']==period and r['variant']==c and r['group']=='all') for c in CASES]
        ax.bar(['1+2','1+3','2+3','1+2+3'],values)
        for i,v in enumerate(values): ax.text(i,v,f'{v:.4f}',ha='center',va='bottom')
        ax.set(title=period,ylabel='Points MAE',ylim=(0,max(values)*1.15))
    fig.tight_layout(); fig.savefig(out/'mae_combinations.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    main()
