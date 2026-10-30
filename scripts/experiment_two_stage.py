"""Fixed 8-fit exploratory minutes x scoring-rate experiment, development only."""
import argparse
import json
import platform
import importlib.metadata
from datetime import datetime,timezone
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.build_features_baseline import read_csv,save_csv,sha
from scripts.compare_models import metrics,validate_development_rows
from scripts.experiment_recent_changes import ROOT,development_history,augment_development_rows,group_scores
from scripts.two_stage_model import TwoStageModel,attach_minute_targets,FEATURES
from scripts.time_validation_tuning import build_folds,FitBudget,stratified_interval,dump,FOLDS

VARIANTS=['baseline','v1','v2','two_stage']


def control_predictions(rows,controls,column):
    index={}
    for r in controls:
        key=r['player_id'],r['game_id']
        if key in index:
            raise ValueError('Duplicate control prediction')
        index[key]=r
    if len(rows)!=len(index) or {(r['player_id'],r['game_id']) for r in rows}!=set(index):
        raise ValueError('Different control cohort')
    result=[]
    for row in rows:
        ref=index[(row['player_id'],row['game_id'])]
        if float(row['target_pts'])!=float(ref['target_pts']):
            raise ValueError('Different control target')
        value=float(ref[column])
        if not np.isfinite(value):
            raise ValueError('Nonfinite control')
        result.append(value)
    return np.asarray(result)


def minute_groups(rows):
    m=np.asarray([float(r['target_minutes']) for r in rows])
    base=np.asarray([float(r['min_mean_5']) for r in rows])
    return {'under10':m<10,'10_to_under20':(m>=10)&(m<20),'20_to_under30':(m>=20)&(m<30),
            '30_plus':m>=30,'minutes_change_10_plus':np.abs(m-base)>=10}


def save_rows(path,rows):
    save_csv(path,rows,list(rows[0]))


def compute_scores(period,rows):
    y=np.asarray([float(r['target_pts']) for r in rows]); minutes=np.asarray([float(r['target_minutes']) for r in rows])
    baseline=np.asarray([float(r['pred_baseline']) for r in rows]); scores=[]; minute_scores=[]; diagnostics=[]
    for variant in VARIANTS:
        pred=np.asarray([float(r['pred_'+variant]) for r in rows])
        scores.extend(dict(period=period,variant=variant,**s) for s in group_scores(y,pred,baseline))
        for group,mask in minute_groups(rows).items():
            score=metrics(y[mask],pred[mask]) if mask.any() else dict(n=0,mae=None,rmse=None)
            diagnostics.append(dict(period=period,variant=variant,group=group,**score))
    for variant,column in [('last5_mean','min_mean_5'),('predicted_minutes','pred_minutes')]:
        minute_scores.append(dict(period=period,variant=variant,**metrics(minutes,[float(r[column]) for r in rows])))
    # A diagnostic with actual minutes is never included in the forecast comparison.
    oracle=np.asarray([float(r['diagnostic_actual_minutes_points']) for r in rows])
    oracle_scores=[dict(period=period,diagnostic='actual_minutes_times_pred_rate',**s) for s in group_scores(y,oracle,baseline)]
    return scores,minute_scores,diagnostics,oracle_scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/week2/two_stage_v3')
    out=parser.parse_args().out
    if out.exists():
        raise ValueError('Output exists; never overwrite a completed experiment')
    old=ROOT/'outputs/week2/time_validation_tuning'; recent=ROOT/'outputs/week2/recent_change_v2'
    raw=ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv'
    train_path=ROOT/'outputs/week1/team_lineup_features/train.csv'; val_path=train_path.with_name('validation.csv')
    inputs=[raw,train_path,val_path,old/'common_parameters.json',old/'predictions_time_validation.csv',old/'predictions_february.csv',
            recent/'predictions_CV_pooled.csv',recent/'predictions_February.csv',recent/'validation.json',
            ROOT/'outputs/week1/features_baseline/validation.json',old/'validation.json']
    # Hashing the old final-test artifacts protects them; their outcomes are never parsed.
    protected=[p for folder in [old,recent,ROOT/'outputs/week2/final_test'] for p in folder.rglob('*') if p.is_file()]
    protected += [ROOT/'nba_dashboard/data.py',ROOT/'nba_dashboard/views.py',ROOT/'streamlit_app.py',
                  ROOT/'requirements.txt',ROOT/'requirements-modeling.txt']
    before={str(p.relative_to(ROOT)):sha(p) for p in set(inputs+protected)}
    v2_meta=json.loads((recent/'validation.json').read_text(encoding='utf-8'))
    for name in ['predictions_CV_pooled.csv','predictions_February.csv']:
        if sha(recent/name)!=v2_meta['output_sha256'][name]:
            raise ValueError('Changed v2 control artifact')
    raw_meta=json.loads(inputs[-2].read_text(encoding='utf-8'))
    from nba_dashboard.data import expected_hash
    if sha(raw)!=expected_hash(raw_meta['source_sha256'],raw.relative_to(ROOT)):
        raise ValueError('Raw source differs from original feature source')
    history=development_history(read_csv(raw))
    train0,val0=read_csv(train_path),read_csv(val_path); validate_development_rows(train0,val0)
    train=attach_minute_targets(augment_development_rows(train0,history),history)
    val=attach_minute_targets(augment_development_rows(val0,history),history)
    for rows in [train,val]:
        rows.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    if (len(train),len(val))!=(13281,3462):
        raise ValueError('Development cohort changed')
    folds=build_folds(train)
    params=json.loads((old/'common_parameters.json').read_text(encoding='utf-8'))['selected']['xgboost']['params']
    cv_control=read_csv(recent/'predictions_CV_pooled.csv'); feb_control=read_csv(recent/'predictions_February.csv')
    original_cv=read_csv(old/'predictions_time_validation.csv'); original_feb=read_csv(old/'predictions_february.csv')
    out.mkdir(parents=True); (out/'models').mkdir(); (out/'features').mkdir()
    config=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),parameters=params,features=FEATURES,
                minute_target='MIN',rate_target='PTS/MIN',rate_sample_weight='training_MIN / training_mean_MIN',
                prediction='max(pred_minutes,0) * max(pred_rate,0)',minute_upper_cap=None,fit_cap=8,
                folds=FOLDS,train_n=len(train),february_n=len(val),test_scoring_allowed=False,
                exploratory_after_test_observed=True,development_reused=True,auto_promotion=False,
                bootstrap_iterations=2000,seed=42,diagnostic_actual_minutes_is_not_forecast=True)
    dump(out/'experiment_config.json',config)
    save_rows(out/'features/train.csv',train); save_rows(out/'features/validation.csv',val)
    budget=FitBudget(8); predictions={}; fits=[]; reload_verified=True
    with threadpool_limits(limits=2):
        for period,tr,va in folds+[('February',train,val)]:
            cv=period!='February'
            controls=[r for r in cv_control if r['period']==period] if cv else feb_control
            original=[r for r in original_cv if r['fold']==period] if cv else original_feb
            v1=control_predictions(va,controls,'pred_v1'); v2=control_predictions(va,controls,'pred_v2')
            original_pred=control_predictions(va,original,'pred_xgboost_02_starter' if cv else 'pred_xgboost_starter')
            if not np.array_equal(v1,original_pred):
                raise ValueError('V1 controls differ from original frozen model')
            if max(r['GAMEDATE'] for r in tr)>=min(r['GAMEDATE'] for r in va):
                raise ValueError('Training overlaps evaluation dates')
            model=TwoStageModel(params).fit(tr,budget)
            pred=model.predict_components(va)
            # Predictions must be identical if target labels are absent or falsified.
            without_labels=[{k:v for k,v in r.items() if not k.startswith('target_')} for r in va]
            if not np.array_equal(pred['points'],model.predict_components(without_labels)['points']):
                raise ValueError('Outcome leakage at prediction boundary')
            modelpath=out/'models'/f'{period}.joblib'; joblib.dump(model,modelpath)
            loaded=joblib.load(modelpath).predict_components(without_labels)
            reload_verified &= all(np.array_equal(pred[key],loaded[key]) for key in pred)
            for stage in ['minutes','rate']:
                fits.append(dict(period=period,stage=stage,fit_number=len(fits)+1,n_train=len(tr),n_validation=len(va),
                    train_end=max(r['GAMEDATE'] for r in tr),validation_start=min(r['GAMEDATE'] for r in va)))
            rows=[]
            for i,r in enumerate(va):
                rows.append({k:r[k] for k in ['player_id','game_id','GAMEDATE','target_pts','target_minutes','min_mean_5']}|
                    dict(period=period,pred_baseline=float(r['pts_mean_5']),pred_v1=float(v1[i]),pred_v2=float(v2[i]),
                         pred_two_stage=float(pred['points'][i]),pred_minutes=float(pred['minutes'][i]),
                         pred_rate=float(pred['rate'][i]),raw_pred_minutes=float(pred['raw_minutes'][i]),
                         raw_pred_rate=float(pred['raw_rate'][i]),
                         diagnostic_actual_minutes_points=float(r['target_minutes'])*float(pred['rate'][i])))
            predictions[period]=rows
    cv=sum([predictions[name] for name,_,_ in folds],[]); predictions['CV_pooled']=cv
    if len(cv)!=9023 or len({(r['player_id'],r['game_id']) for r in cv})!=9023:
        raise ValueError('Wrong pooled validation cohort')
    scores=[]; minute_scores=[]; diagnostics=[]; oracle_scores=[]; intervals=[]
    for period,rows in predictions.items():
        a,b,c,d=compute_scores(period,rows); scores+=a; minute_scores+=b; diagnostics+=c; oracle_scores+=d
        if period in ['CV_pooled','February']:
            save_rows(out/f'predictions_{period}.csv',rows)
            for reference in ['v1','v2']:
                result=stratified_interval([r['target_pts'] for r in rows],[r['pred_'+reference] for r in rows],
                    [r['pred_two_stage'] for r in rows],[r['GAMEDATE'] for r in rows],[r['period'] for r in rows])
                intervals.append(dict(period=period,reference=reference,**result))
    for name,rows in [('scores',scores),('minute_scores',minute_scores),('minute_group_scores',diagnostics),
                      ('diagnostic_actual_minutes_scores',oracle_scores),('fit_log',fits)]:
        save_rows(out/f'{name}.csv',rows)
    dump(out/'paired_intervals.json',intervals)
    make_report(out,scores,minute_scores,oracle_scores,intervals)
    if budget.count!=8 or len(fits)!=8 or not reload_verified or any(sha(ROOT/p)!=h for p,h in before.items()):
        raise ValueError('Budget/reload/protected file check failed')
    code=[Path(__file__),ROOT/'scripts/two_stage_model.py',ROOT/'scripts/recent_change_features.py',
          ROOT/'scripts/experiment_recent_changes.py',ROOT/'scripts/time_validation_tuning.py']
    evidence=dict(fits=budget.count,cohort_original_values_equal=True,train_n=len(train),february_n=len(val),cv_n=len(cv),
         controls_exact=True,model_reload_equal=bool(reload_verified),prediction_without_labels_equal=True,
         test_metrics_computed=False,production_promoted=False,source_sha256=before,
         code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},
         environment=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in
                          ['numpy','scikit-learn','xgboost','joblib','matplotlib']}),
         output_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    dump(out/'validation.json',evidence)
    print(json.dumps(dict(output=str(out),fits=budget.count,intervals=intervals),ensure_ascii=False))


def make_report(out,scores,minute_scores,oracle_scores,intervals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=dict(baseline='五場均值',v1='九項直接得分',v2='十四項直接得分',two_stage='預估分鐘×預估效率')
    lines=['# 預估上場時間與得分效率：兩階段研究','',
        '使用同一組十四項嚴格前日特徵分別預估分鐘與每分鐘得分，最後相乘。效率訓練以正分鐘作權重，短出賽不刪除，零分出賽保留。',
        '沿用原固定XGBoost設定，三輪加二月共8次fit。九項及十四項對照使用既有凍結預測，沒有重訓對照或追加調參。','',
        '|期間|樣本|模型|MAE|RMSE|','|---|---:|---|---:|---:|']
    for period in ['F1','F2','F3','CV_pooled','February']:
        for variant in VARIANTS:
            s=next(r for r in scores if r['period']==period and r['variant']==variant and r['group']=='all')
            lines.append(f"|{period}|{s['n']}|{names[variant]}|{s['mae']:.4f}|{s['rmse']:.4f}|")
    lines+=['','## 分鐘預估','', '|期間|方法|分鐘MAE|分鐘RMSE|','|---|---|---:|---:|']
    for r in minute_scores:
        if r['period'] in ['CV_pooled','February']:
            lines.append(f"|{r['period']}|{r['variant']}|{r['mae']:.4f}|{r['rmse']:.4f}|")
    lines+=['','## 高得分場次','', '|期間|實際得分組|樣本|模型|MAE|RMSE|','|---|---|---:|---|---:|---:|']
    for r in scores:
        if r['period'] in ['CV_pooled','February'] and r['group']!='all':
            lines.append(f"|{r['period']}|{r['group']}|{r['n']}|{names[r['variant']]}|{r['mae']:.4f}|{r['rmse']:.4f}|")
    lines+=['','## 成對差異','', '差異為兩階段減對照，負數代表改善；95%區間按日期群組重抽2000次，CV在每輪內抽日期。']
    for r in intervals:
        lines.append(f"- {r['period']}／{names[r['reference']]}：MAE差 {r['delta_mae']:.4f}，95%區間 [{r['lower_95']:.4f}, {r['upper_95']:.4f}]。")
    lines+=['','## 已知實際分鐘的事後診斷','',
            '以下使用已知的當場實際分鐘乘預估效率，只協助分辨分鐘誤差與效率問題；不是可用的賽前預測，也不是保證能達到的準確度上限。']
    for r in oracle_scores:
        if r['period'] in ['CV_pooled','February'] and r['group']=='all':
            lines.append(f"- {r['period']}：MAE {r['mae']:.4f}，RMSE {r['rmse']:.4f}。")
    lines+=['','## 限制與建議','',
        '本研究在看過舊測試後提出，並重用已有開發資料，屬探索結果。三月至四月不產生新預測、不再測試；正式模型與Streamlit維持九項。',
        '歷史先發為賽後快照，仍稱日期隔離的假設性回測；結果以球員有上場為前提，不能預測是否出賽。',
        '效率模型學的是按分鐘加權的得分率，不是每場得分率的簡單平均。兩個模型都存在估計誤差，相乘後仍可能放大偏差；拆開學習不保證更準。',
        '分鐘區間與分鐘變化10分鐘以上群組見 minute_group_scores.csv，都是事後診斷，不作賽前分組或選模。',
        '建議依整體MAE／RMSE與高分群組一起判讀，保留原正式模型，任何升級須先在未檢視的新資料另定驗證。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,period in zip(axes,['CV_pooled','February']):
        values=[next(r['mae'] for r in scores if r['period']==period and r['variant']==v and r['group']=='all') for v in VARIANTS]
        ax.bar(['Baseline','9 direct','14 direct','Two stage'],values)
        for i,v in enumerate(values):
            ax.text(i,v,f'{v:.4f}',ha='center',va='bottom')
        ax.set(title=period,ylabel='Points MAE',ylim=(0,max(values)*1.15))
    fig.tight_layout(); fig.savefig(out/'points_mae_comparison.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    main()
