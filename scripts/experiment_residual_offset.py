"""Eight fixed offset fits; development comparison plus separate reused Luka demo."""
import json
from pathlib import Path
from datetime import datetime,timezone
import platform
import importlib.metadata

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.build_features_baseline import read_csv,sha
from scripts.compare_models import validate_development_rows
from scripts.experiment_recent_changes import ROOT
from scripts.experiment_two_stage import save_rows,control_predictions
from scripts.high_score_weighting import point_group_scores,alert_metrics,paired_comparison
from scripts.residual_offset import ResidualOffsetModel,COLUMNS
from scripts.same_conference_h2h import h2h_matrix
from scripts.time_validation_tuning import build_folds,FitBudget,dump,FOLDS
from nba_dashboard.data import expected_hash

VARIANTS={'residual_w1':1,'residual_w2':2}
ALL=['baseline','direct_w1','residual_w1','direct_w2','residual_w2']
PAIRS=[('baseline','residual_w1'),('direct_w1','residual_w1'),('direct_w2','residual_w2'),('residual_w1','residual_w2')]
NAMES=dict(baseline='五場均值',direct_w1='直接得分20項',residual_w1='均值＋增減',
           direct_w2='直接得分20項／高分2倍',residual_w2='均值＋增減／高分2倍')


def summarize(period,rows):
    scores=[]; alerts=[]
    for variant in ALL:
        y=[r['target_pts'] for r in rows]; p=[r['pred_'+variant] for r in rows]
        scores.extend(dict(period=period,variant=variant,**s) for s in point_group_scores(y,p))
        alerts.append(dict(period=period,variant=variant,**alert_metrics(y,p)))
    return scores,alerts


def main():
    out=ROOT/'outputs/week2/residual_offset_v8'
    if out.exists(): raise ValueError('Output already exists; no overwrite or extra fits')
    features_dir=ROOT/'outputs/week2/same_conference_h2h_v6'
    previous=ROOT/'outputs/week2/high_score_weighting_v7'
    old_demo=ROOT/'outputs/demos/luka_combined_w2_march_2026'
    inputs=[features_dir/'features/train.csv',features_dir/'features/validation.csv',
        previous/'predictions_CV_pooled.csv',previous/'predictions_February.csv',previous/'validation.json',
        ROOT/'outputs/week2/time_validation_tuning/common_parameters.json',old_demo/'features.csv',
        old_demo/'predictions.csv',old_demo/'validation.json']
    protected=[p for p in (ROOT/'outputs/week2').rglob('*') if p.is_file()]+[p for p in old_demo.iterdir() if p.is_file()]
    protected += [ROOT/'nba_dashboard/data.py',ROOT/'nba_dashboard/views.py',ROOT/'streamlit_app.py']
    before={str(p.relative_to(ROOT)):sha(p) for p in set(inputs+protected)}
    meta=json.loads(inputs[4].read_text(encoding='utf-8'))
    for p in inputs[:2]:
        if sha(p)!=expected_hash(meta['source_sha256'],p.relative_to(ROOT)): raise ValueError('Changed source features')
    for p in inputs[2:4]:
        if sha(p)!=expected_hash(meta['output_sha256'],p.relative_to(previous)): raise ValueError('Changed direct controls')
    demo_meta=json.loads(inputs[8].read_text(encoding='utf-8'))
    for p in inputs[6:8]:
        if sha(p)!=expected_hash(demo_meta['output_sha256'],p.relative_to(old_demo)): raise ValueError('Changed Luka source')
    train,val=[read_csv(p) for p in inputs[:2]]; validate_development_rows(train,val)
    for rows in [train,val]: rows.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    if (len(train),len(val))!=(13221,3462): raise ValueError('Changed common cohort')
    folds=build_folds(train)
    if [(len(tr),len(va)) for _,tr,va in folds]!=[(4198,4104),(8302,2422),(10724,2497)]: raise ValueError('Wrong folds')
    params=json.loads(inputs[5].read_text(encoding='utf-8'))['selected']['xgboost']['params']
    cv_controls=read_csv(inputs[2]); feb_controls=read_csv(inputs[3])
    out.mkdir(parents=True); (out/'models').mkdir()
    dump(out/'experiment_config.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),params=params,
        columns=COLUMNS,variants=VARIANTS,fit_cap=8,folds=FOLDS,common_train_n=13221,cv_n=9023,february_n=3462,
        training_target='target_pts - pts_mean_5',prediction='max(0,pts_mean_5 + predicted_delta)',
        weights_based_on_original_points=True,evaluation_unweighted=True,pairs=PAIRS,paired_groups=['all','30+'],
        bootstrap_iterations=2000,seed=42,multiplicity_adjusted=False,auto_promotion=False,
        development_reused=True,exploratory_after_test_observed=True,
        luka_scope='continuation of user-requested same 17-game reused-data illustration only; no selection',
        league_wide_march_scoring_allowed=False))
    targets=[dict(player_id=r['player_id'],game_id=r['game_id'],GAMEDATE=r['GAMEDATE'],split=r['split'],
        actual_pts=float(r['target_pts']),pts_mean_5=float(r['pts_mean_5']),
        target_delta=float(r['target_pts'])-float(r['pts_mean_5'])) for r in train+val]
    save_rows(out/'training_and_validation_targets.csv',targets)
    predictions={}; fit_log=[]; budget=FitBudget(8); reload_equal=True
    with threadpool_limits(limits=2):
        for period,tr,va in folds+[('February',train,val)]:
            controls=[r for r in cv_controls if r['period']==period] if period!='February' else feb_controls
            d1=control_predictions(va,controls,'pred_combined_unweighted'); d2=control_predictions(va,controls,'pred_combined_w2')
            rows=[{k:r[k] for k in ['player_id','game_id','GAMEDATE','target_pts']}|
                dict(period=period,pred_baseline=float(r['pts_mean_5']),pred_direct_w1=float(d1[i]),pred_direct_w2=float(d2[i])) for i,r in enumerate(va)]
            no_labels=[{k:v for k,v in r.items() if not k.startswith('target_')} for r in va]
            for variant,multiplier in VARIANTS.items():
                model=ResidualOffsetModel(params,multiplier).fit(tr,budget)
                result=model.predict_components(va)
                changed=model.predict_components([dict(r,target_pts=99999,target_delta=99999,PTS=99999,MIN=99999) for r in no_labels])
                if any(not np.array_equal(result[k],changed[k]) for k in result): raise ValueError('Outcome-dependent offset inference')
                path=out/'models'/f'{period}_{variant}.joblib'; joblib.dump(model,path)
                again=joblib.load(path).predict_components(no_labels)
                reload_equal &= all(np.array_equal(result[k],again[k]) for k in result)
                for i,r in enumerate(rows):
                    r['pred_'+variant]=float(result['points'][i]); r[variant+'_delta']=float(result['delta'][i]); r[variant+'_raw_points']=float(result['raw_points'][i])
                fit_log.append(dict(period=period,variant=variant,fit_number=budget.count,n_train=len(tr),n_validation=len(va),
                    multiplier=multiplier,raw_weight_normalizer=model.weight_normalizer,train_high_n=model.train_high_n,
                    mean_target_delta=float(np.mean([float(r['target_pts'])-float(r['pts_mean_5']) for r in tr])),
                    train_end=model.trained_through,validation_start=min(r['GAMEDATE'] for r in va)))
            predictions[period]=rows
    cv=sum([predictions[name] for name,_,_ in folds],[]); predictions['CV_pooled']=cv
    if len(cv)!=9023: raise ValueError('Wrong pooled cohort')
    scores=[]; alerts=[]; intervals=[]
    for period,rows in predictions.items():
        s,a=summarize(period,rows); scores+=s; alerts+=a
        if period in ['CV_pooled','February']:
            save_rows(out/f'predictions_{period}.csv',rows)
            for first,second in PAIRS:
                for group in ['all','30+']:
                    intervals.append(dict(period=period,first=first,second=second,group=group,**paired_comparison(rows,first,second,group)))
    save_rows(out/'scores.csv',scores); save_rows(out/'alert_metrics.csv',alerts); save_rows(out/'fit_log.csv',fit_log)
    dump(out/'paired_intervals.json',intervals)
    references=[r for r in read_csv(previous/'reference_scores.csv') if r['variant']=='frozen9']
    save_rows(out/'reference_scores.csv',references)
    make_report(out,scores,references)
    # Only after all fixed development fits, reuse the existing authorized Luka illustration.
    run_luka_demo(out,old_demo,features_dir)
    if budget.count!=8 or not reload_equal or any(sha(ROOT/p)!=h for p,h in before.items()): raise ValueError('Budget/reload/protected source changed')
    code=[Path(__file__),ROOT/'scripts/residual_offset.py',ROOT/'scripts/high_score_weighting.py',ROOT/'scripts/same_conference_h2h.py',ROOT/'scripts/time_validation_tuning.py']
    dump(out/'validation.json',dict(fits=8,common_train_n=13221,cv_n=9023,february_n=3462,
        delta_target_verified=True,weights_from_original_points=True,model_reload_equal=bool(reload_equal),
        production_promoted=False,independent_test=False,luka_reused_demo_n=17,league_wide_march_predictions=False,
        source_sha256=before,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},
        environment=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['numpy','scikit-learn','xgboost','joblib','matplotlib']}),
        output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(dict(output=str(out),fits=8,summary=[s for s in scores if s['period'] in ['CV_pooled','February'] and s['group'] in ['all','30+']]),ensure_ascii=False))


def run_luka_demo(out,old_demo,features_dir):
    demo=out/'luka_demo'; demo.mkdir()
    features=read_csv(old_demo/'features.csv'); old=read_csv(old_demo/'predictions.csv')
    index={(r['player_id'],r['game_id']):r for r in old}
    if len(features)!=17 or len(index)!=17 or {(r['player_id'],r['game_id']) for r in features}!=set(index): raise ValueError('Changed fixed Luka cohort')
    if any(r['player_id']!='Luka Dončić' or not '2026-03-01'<=r['GAMEDATE']<='2026-04-12' for r in features): raise ValueError('Unauthorized demo scope')
    dump(demo/'demo_protocol.json',dict(scope='same user-requested 17-game reused-data illustration',independent_test=False,
        fits=0,models_trained_through='2026-01-31',selection_allowed=False,prior_demo_unchanged=True))
    direct=joblib.load(features_dir/'models/February_case13_h2h20.joblib')
    dp=direct['model'].predict(h2h_matrix(features,direct['columns'])).astype(float)
    result={v:joblib.load(out/'models'/f'February_{v}.joblib').predict_components(features) for v in VARIANTS}
    rows=[]
    for i,f in enumerate(features):
        r=index[(f['player_id'],f['game_id'])]
        rows.append(dict(player_id=r['player_id'],game_id=r['game_id'],GAMEDATE=r['GAMEDATE'],MATCHUP=r['MATCHUP'],
            actual_minutes=float(r['actual_minutes']),target_pts=float(r['actual_pts']),pred_baseline=float(r['pred_baseline']),
            pred_direct_w1=float(dp[i]),pred_direct_w2=float(r['pred_weighted']),
            pred_residual_w1=float(result['residual_w1']['points'][i]),pred_residual_w2=float(result['residual_w2']['points'][i]),
            residual_w1_delta=float(result['residual_w1']['delta'][i]),residual_w2_delta=float(result['residual_w2']['delta'][i])))
    save_rows(demo/'predictions.csv',rows)
    human=[dict(日期=r['GAMEDATE'],對戰=r['MATCHUP'],實際上場分鐘=r['actual_minutes'],實際得分=r['target_pts'],
        五場均值=round(r['pred_baseline'],2),預估增減=round(r['residual_w1_delta'],2),均值加增減=round(r['pred_residual_w1'],2),
        高分加權增減=round(r['residual_w2_delta'],2),均值加增減2倍=round(r['pred_residual_w2'],2),舊直接2倍=round(r['pred_direct_w2'],2)) for r in rows]
    save_rows(demo/'Luka_均值增減預測.csv',human)
    s,a=summarize('Luka_reused_demo',rows); save_rows(demo/'scores.csv',s); save_rows(demo/'alert_metrics.csv',a)
    references=[dict(period='Luka_reused_demo',variant='frozen9',**g) for g in
        point_group_scores([float(r['actual_pts']) for r in old],[float(r['pred_original']) for r in old])]
    save_rows(demo/'reference_scores.csv',references)
    make_report(demo,s,references)
    plot_luka(demo,rows)


def make_report(out,scores,references=None):
    lines=['# 預測相對近五場均值的增減','',
        '最終預測＝近五場均值＋模型預估增減，只將最後得分裁至非負。增減可正可負；高分2倍權重根據原實際得分>=30，不根據增減幅度。',
        '同一20項特徵與原固定XGBoost設定，兩版本×四期共8fit，不調參。對照資料、原成果與線上模型不變。','',
        '|期間|模型|整體MAE|RMSE|30+ MAE|<30 MAE|','|---|---|---:|---:|---:|---:|']
    periods=[p for p in ['CV_pooled','February','Luka_reused_demo'] if any(s['period']==p for s in scores)]
    for period in periods:
        for variant in ALL:
            get=lambda g:next(s for s in scores if s['period']==period and s['variant']==variant and s['group']==g)
            all_,high,low=[get(g) for g in ['all','30+','under30']]
            lines.append(f"|{period}|{NAMES[variant]}|{all_['mae']:.4f}|{all_['rmse']:.4f}|{high['mae']:.4f}|{low['mae']:.4f}|")
    lines+=['','所有指標是最終得分的未加權誤差，不是對增減標籤評分，也沒有只挑高分場次。完整分箱／警示與AP另存CSV。',
        '近五場均值以嚴格前日正分鐘計算，首次同區對陣缺失仍為NaN。先發為賽後快照，屬日期隔離的假設性回測。',
        'CV／二月為已重用開發資料；Luka為延續使用者要求的同17場已使用資料展示，模型固定一月底、不依展示選模。不是獨立新測試，不全聯盟重評三月，不改正式test_usage或線上。',
        '若增減模型不優於五場均值，表示這次特徵與設定的修正沒有提供增益；不能因保留基準就假定一定更準。']
    if references:
        lines+=['','原九項凍結模型另列作背景參考（訓練13281筆，研究共同訓練13221筆）：',
                '', '|期間|整體MAE|RMSE|30+ MAE|','|---|---:|---:|---:|']
        for period in periods:
            group={r['group']:r for r in references if r['period']==period}
            if group:
                lines.append(f"|{period}|{float(group['all']['mae']):.4f}|{float(group['all']['rmse']):.4f}|{float(group['30+']['mae']):.4f}|")
    lines+=['','本次結論：CV與二月的增減模型均未勝過同權重的20項直接得分模型。Luka同17場的2倍增減模型MAE為7.99，低於舊直接2倍的9.20，但仍高於五場均值7.49；單一球員展示不足以支持換模。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot_luka(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5)); x=np.arange(len(rows))
    for col,label in [('target_pts','Actual'),('pred_baseline','Last5 baseline'),('pred_residual_w1','Baseline + delta'),
                      ('pred_residual_w2','Baseline + delta / 2x'),('pred_direct_w2','Previous direct / 2x')]:
        ax.plot(x,[r[col] for r in rows],marker='o',label=label)
    ax.set(xticks=x,xticklabels=[r['GAMEDATE'][5:] for r in rows],ylabel='Points',title='Luka Doncic | baseline plus predicted delta | reused-data illustration',ylim=(0,max(r['target_pts'] for r in rows)+5))
    ax.tick_params(axis='x',rotation=45); ax.legend(fontsize=9); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(out/'luka_offset_predictions.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    main()
