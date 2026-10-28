"""One frozen final evaluation, including predeclared high-score diagnostics."""
import argparse
import json
import math
from pathlib import Path
from datetime import date,datetime,timezone

import numpy as np
import joblib
from threadpoolctl import threadpool_limits

if __package__:
    from .analyze_residuals import summarize,grouped_bands,player_profiles,BANDS,BAND_LABELS
    from .time_validation_tuning import stratified_interval
    from .build_features_baseline import read_csv,save_csv,sha
else:
    from analyze_residuals import summarize,grouped_bands,player_profiles,BANDS,BAND_LABELS
    from time_validation_tuning import stratified_interval
    from build_features_baseline import read_csv,save_csv,sha

ROOT=Path(__file__).resolve().parents[1]
GROUPS=['basic','starter','teammates']


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def validate_test_rows(rows):
    seen=set()
    if not rows:
        raise ValueError('Empty test cohort')
    for r in rows:
        day=r['GAMEDATE']; key=r['player_id'],r['game_id']
        actual=float(r['target_pts'])
        if (date.fromisoformat(day).isoformat()!=day or not '2026-03-01'<=day<='2026-04-12'
            or r['split']!='test' or key in seen or not r['game_id'].endswith('|'+day)
            or not math.isfinite(actual) or actual<0):
            raise ValueError('Invalid test date, key, split, or outcome')
        seen.add(key)


def frozen_model_check(bundle,columns):
    if bundle.get('fit_end_date')!='2026-01-31' or bundle.get('feature_columns')!=columns:
        raise ValueError('Model cutoff or feature order differs from the frozen protocol')


def evaluation_rows(records,predictions):
    predictions=np.asarray(predictions,dtype=float)
    if predictions.ndim!=1 or len(predictions)!=len(records) or not np.isfinite(predictions).all():
        raise ValueError('Predictions are missing, unaligned, or nonfinite')
    result=[]
    for r,pred in zip(records,predictions,strict=True):
        actual=float(r['target_pts']); baseline=float(r['pts_mean_5'])
        if not math.isfinite(actual) or not math.isfinite(baseline):
            raise ValueError('Nonfinite outcome or baseline')
        result.append(dict(player_id=r['player_id'],game_id=r['game_id'],GAMEDATE=r['GAMEDATE'],
            actual=actual,predicted=float(pred),baseline=baseline,residual=float(actual-pred),
            absolute_error=float(abs(actual-pred)),baseline_residual=actual-baseline,
            baseline_absolute_error=abs(actual-baseline),is_starter=int(float(r['is_starter'])),
            teammate_unknown_count=int(float(r['teammate_unknown_count']))))
    return result


def high_score_groups(rows):
    return [dict(group=label,**summarize(data)) for label,data in [
        ('all',rows),('actual_20_plus',[r for r in rows if r['actual']>=20]),
        ('actual_30_plus',[r for r in rows if r['actual']>=30])]]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/week2/final_test')
    args=parser.parse_args(); out=args.output_dir
    if out.exists():
        raise FileExistsError('A final test run already exists; do not overwrite or silently retest')
    source=ROOT/'outputs/week2/time_validation_tuning'
    candidate_path=source/'selected_candidate.json'
    candidate=json.loads(candidate_path.read_text(encoding='utf-8'))
    encoder_path=source/'encoders/February.json'
    encoding=json.loads(encoder_path.read_text(encoding='utf-8'))
    protected=[p for p in source.rglob('*') if p.is_file()]
    protected+=[p for p in (ROOT/'outputs/week2/residual_analysis').iterdir() if p.is_file()]
    frozen_sources={str(p.relative_to(ROOT)):sha(p) for p in protected}
    old_validation=json.loads((source/'validation.json').read_text(encoding='utf-8'))
    assert all(sha(source/p)==h for p,h in old_validation['output_sha256'].items())
    bundles={}; model_manifest=[]
    primary=candidate['result']['model']+'_'+candidate['result']['features']
    for name in ['linear','tree','xgboost']:
        for group in GROUPS:
            key=name+'_'+group; path=source/'models'/(key+'.joblib')
            bundle=joblib.load(path)
            frozen_model_check(bundle,encoding['feature_columns'][group])
            if group=='teammates' and bundle['encoder']!=encoding['encoder']:
                raise ValueError('Persisted encoder changed')
            bundles[key]=bundle
            model_manifest.append(dict(model_variant=key,role='primary' if key==primary else 'secondary',
                model_file=str(path.relative_to(ROOT)),sha256=sha(path),feature_columns=bundle['feature_columns'],params=bundle['params']))
    assert sha(source/candidate['model_file'])==candidate['model_sha256']
    assert primary in bundles
    out.mkdir(parents=True)
    # This protocol is written before either test CSV is opened or hashed.
    protocol=dict(version='frozen-final-test-v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        primary_model=primary,baseline='pts_mean_5',models=model_manifest,
        candidate_sha256=sha(candidate_path),encoder_sha256=sha(encoder_path),
        test_dates=['2026-03-01','2026-04-12'],expected_eligible_samples=6979,
        high_score_thresholds=[20,30],high_score_groups_are_nested=True,
        metric_order=['mae','rmse'],residual_definition='actual_minus_prediction',
        date_cluster_bootstrap_iterations=2000,seed=42,refit=False,model_reselection=False,
        test_inputs=['outputs/week1/teammate_encoding/test.csv','outputs/week1/team_lineup_features/test.csv'],
        fixed_player_cases='outputs/week2/residual_analysis/case_selection.csv',frozen_source_sha256=frozen_sources)
    dump(out/'test_protocol.json',protocol)
    protocol_sha=sha(out/'test_protocol.json')
    test_paths=[ROOT/p for p in protocol['test_inputs']]
    test_hashes={str(p.relative_to(ROOT)):sha(p) for p in test_paths}
    dump(out/'test_input_manifest.json',dict(protocol_sha256=protocol_sha,source_sha256=test_hashes))
    dump(out/'test_usage.json',dict(status='test_access_in_progress',protocol_sha256=protocol_sha,
                                   test_source_sha256=test_hashes,model_reselection=False))
    records=read_csv(test_paths[0]); original=read_csv(test_paths[1])
    validate_test_rows(records); validate_test_rows(original)
    assert len(records)==len(original)==6979
    reference={(r['player_id'],r['game_id']):r for r in original}
    assert set(reference)=={(r['player_id'],r['game_id']) for r in records}
    for r in records:
        old=reference[(r['player_id'],r['game_id'])]
        assert r['GAMEDATE']==old['GAMEDATE']
        assert all(float(r[c])==float(old[c]) for c in encoding['feature_columns']['starter']+['target_pts'])
        assert r['pregame_verified']=='False'
        flags=[float(r[c]) for c in encoding['encoder']['feature_names'][:-1]]
        unknown=float(r['teammate_unknown_count']); starter=float(r['is_starter'])
        assert all(f in [0,1] for f in flags) and unknown>=0 and unknown.is_integer() and starter in [0,1]
        assert sum(flags)+unknown==5-starter
        if r['player_id'] in encoding['encoder']['classes']:
            assert flags[encoding['encoder']['classes'].index(r['player_id'])]==0
    records.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    predictions={'baseline':np.asarray([float(r['pts_mean_5']) for r in records])}
    with threadpool_limits(limits=2):
        for key,bundle in bundles.items():
            X=np.asarray([[float(r[c]) for c in bundle['feature_columns']] for r in records])
            if not np.isfinite(X).all():
                raise ValueError('Missing input; test samples cannot be silently removed')
            predictions[key]=bundle['model'].predict(X)
    all_rows={key:evaluation_rows(records,pred) for key,pred in predictions.items()}
    primary_rows=all_rows[primary]; main_stats=summarize(primary_rows)
    scores=[dict(model_variant=key,role='baseline' if key=='baseline' else 'primary' if key==primary else 'secondary',
                 **summarize(rows)) for key,rows in all_rows.items()]
    save_csv(out/'model_metrics.csv',scores,list(scores[0]))
    exported=[]
    for i,r in enumerate(records):
        row={k:r[k] for k in ['player_id','game_id','GAMEDATE','target_pts','teammate_unknown_count']}
        row.update({'pred_'+key:float(p[i]) for key,p in predictions.items()})
        exported.append(row)
    save_csv(out/'predictions_test.csv',exported,list(exported[0]))
    save_csv(out/'residuals_primary.csv',primary_rows,list(primary_rows[0]))
    high=high_score_groups(primary_rows)
    save_csv(out/'high_score_metrics.csv',high,list(high[0]))
    secondary_groups=[dict(model_variant=key,**g) for key,data in all_rows.items() for g in high_score_groups(data)]
    save_csv(out/'all_models_high_score_metrics.csv',secondary_groups,list(secondary_groups[0]))
    bands=grouped_bands(primary_rows,'actual')+grouped_bands(primary_rows,'predicted')
    save_csv(out/'score_band_metrics.csv',bands,list(bands[0]))
    paired=[]
    for group in high:
        subset=primary_rows if group['group']=='all' else [r for r in primary_rows if r['actual']>=int(group['group'].split('_')[1])]
        if len({r['GAMEDATE'] for r in subset})>=2:
            paired.append(dict(group=group['group'],n=len(subset),**stratified_interval(
                [r['actual'] for r in subset],[r['baseline'] for r in subset],[r['predicted'] for r in subset],
                [r['GAMEDATE'] for r in subset],['test']*len(subset))))
    save_csv(out/'paired_differences.csv',paired,list(paired[0]))
    players=player_profiles(primary_rows)
    for p in players:
        p.pop('case_eligible')
    save_csv(out/'player_metrics.csv',players,list(players[0]))
    fixed_cases=read_csv(ROOT/protocol['fixed_player_cases'])
    by_player={p['player_id']:p for p in players}
    case_results=[dict(role=c['role'],**by_player.get(c['player_id'],dict(player_id=c['player_id'],**summarize([])))) for c in fixed_cases]
    save_csv(out/'validation_selected_player_cases.csv',case_results,list(case_results[0]))
    high_errors=sorted([r for r in primary_rows if r['actual']>=20],key=lambda r:(-r['absolute_error'],r['GAMEDATE'],r['player_id']))[:20]
    if high_errors:
        save_csv(out/'largest_high_score_errors.csv',high_errors,list(high_errors[0]))
    usage=dict(status='test_consumed_final_evaluation',protocol_sha256=protocol_sha,test_source_sha256=test_hashes,
               n=6979,primary_model=primary,model_fits_performed=0,model_reselected=False,
               note='This test is now observed; do not reuse it to tune or select a new model as an unseen holdout.')
    dump(out/'test_usage.json',usage)
    evaluated=dict(candidate,status='final_test_evaluated',test_evaluated=True,
                   original_candidate=str(candidate_path.relative_to(ROOT)),
                   path_base='project_root',
                   model_file=str((source/candidate['model_file']).relative_to(ROOT)),
                   validation_sample_manifest=str((source/candidate['validation_sample_manifest']).relative_to(ROOT)),
                   training_sample_manifest=str((source/candidate['training_sample_manifest']).relative_to(ROOT)),
                   experiment_config=str((source/candidate['experiment_config']).relative_to(ROOT)),
                   encoder_file=str((source/candidate['encoder_file']).relative_to(ROOT)) if candidate['encoder_file'] else None,
                   test_protocol_sha256=protocol_sha,test_metrics=main_stats,
                   test_predictions=str((out/'predictions_test.csv').relative_to(ROOT)),test_predictions_sha256=sha(out/'predictions_test.csv'))
    dump(out/'evaluated_candidate.json',evaluated)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    actual_bands=[b for b in bands if b['grouping']=='actual' and b['n']]
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for j,field in enumerate(['baseline_mae','mae']):
        bars=axs[0].bar(np.arange(len(actual_bands))+(j-.5)*.34,[b[field] for b in actual_bands],.34,label='Last-five mean' if j==0 else 'Locked model')
        axs[0].bar_label(bars,fmt='%.2f',fontsize=8)
    axs[0].set_xticks(range(len(actual_bands)),[BAND_LABELS[BANDS.index(b['band'])]+f'\nn={b["n"]}' for b in actual_bands])
    axs[0].set(xlabel='Actual points (post-outcome grouping)',ylabel='MAE (points)',title='Final test: actual score bands'); axs[0].legend()
    bars=axs[1].bar(range(3),[g['mean_residual'] for g in high],color='#237aab')
    axs[1].bar_label(bars,fmt='%+.2f'); axs[1].axhline(0,color='black',ls='--')
    axs[1].set_xticks(range(3),['All\nn='+str(high[0]['n']),'20+\nn='+str(high[1]['n']),'30+\nn='+str(high[2]['n'])])
    axs[1].set(ylabel='Mean actual minus predicted',title='Nested groups: positive = underprediction')
    fig.savefig(out/'high_score_errors.png',dpi=170); plt.close(fig)
    a=np.array([r['actual'] for r in primary_rows]); p=np.array([r['predicted'] for r in primary_rows])
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    axs[0].scatter(p,a,s=8,alpha=.18,rasterized=True); end=max(a.max(),p.max())+1
    axs[0].plot([0,end],[0,end],'k--'); axs[0].set(xlabel='Predicted points',ylabel='Actual points',title='Final test: n=6,979')
    edges=np.linspace((a-p).min()-1,(a-p).max()+1,41)
    axs[1].hist(a-p,bins=edges,label='All samples',alpha=.7)
    axs[1].hist((a-p)[a>=30],bins=edges,label='Actual 30+',alpha=.8)
    axs[1].axvline(0,color='black',ls='--'); axs[1].set(xlabel='Actual minus predicted',ylabel='Samples',title='Primary model residuals'); axs[1].legend()
    fig.savefig(out/'test_residuals.png',dpi=170); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,metric in zip(axs,['mae','rmse']):
        for j,group in enumerate(GROUPS):
            values=[next(r[metric] for r in scores if r['model_variant']==name+'_'+group) for name in ['linear','tree','xgboost']]
            bars=ax.bar(np.arange(3)+(j-1)*.25,values,.25,label=group); ax.bar_label(bars,fmt='%.3f',fontsize=8)
        ax.axhline(scores[0][metric],color='black',ls='--',label='Last-five mean')
        ax.set_xticks(range(3),['linear','tree','xgboost']); ax.set_ylabel(metric.upper()+' (points)')
        ax.set_title('Frozen comparisons; no test-based reselection'); ax.legend(fontsize=8)
    fig.savefig(out/'frozen_models_comparison.png',dpi=170); plt.close(fig)
    report=['# 最終測試與高得分誤差','',
        '2026-10-23。依事先保存的 test_protocol.json 評分 2026-03-01～2026-04-12 的 6,979 筆合格樣本。主模型維持 XGBoost／starter 九項特徵；九個比較模型均只用一月底以前資料訓練，本次只 predict，沒有重訓、調參、重建類別或依測試排名改選模型。','',
        '## 主模型結果','',
        f'- MAE {main_stats["mae"]:.4f}，RMSE {main_stats["rmse"]:.4f} 分。',
        f'- 五場均值基準 MAE {main_stats["baseline_mae"]:.4f}、RMSE {main_stats["baseline_rmse"]:.4f}；模型 MAE 差異 {main_stats["delta_mae"]:+.4f} 分（{main_stats["delta_mae"]/main_stats["baseline_mae"]*100:+.2f}%）。',
        f'- 誤差不超過5分 {main_stats["within_5_rate"]:.1%}、不超過10分 {main_stats["within_10_rate"]:.1%}；平均殘差 {main_stats["mean_residual"]:+.4f} 分。殘差＝實際－預測，正值為低估。','',
        '## 高得分場次：主要檢查','',
        '20+與30+互相重疊，不能相加。這是按賽後真值分組的描述，不能聲稱預測前已知道哪些比賽會高分。','',
        '| 實際得分組 | 筆數 | 模型 MAE | 基準 MAE | 模型 RMSE | 平均殘差 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for g in high:
        report.append(f'| {g["group"]} | {g["n"]} | {g["mae"]:.3f} | {g["baseline_mae"]:.3f} | {g["rmse"]:.3f} | {g["mean_residual"]:+.3f} |')
    thirty=[r for r in primary_rows if r['actual']>=30]
    if thirty:
        under=sum(r['residual']>0 for r in thirty)
        report+=['',f'30分以上樣本中，{under}/{len(thirty)} 筆被低估（{under/len(thirty):.1%}）。這是實際高分樣本的事後觀察，不代表預測前已可識別該組。']
    report+=['','### 與基準的同日成對差異','',
        'MAE差值＝模型減基準；負值才是改善。日期群組重抽2000次、seed42；以下為探索性區間，沒有處理跨日相依及多重比較，三個重疊組也不是獨立實驗。','',
        '| 分組 | MAE差值 | 探索性95%區間 |','| --- | ---: | --- |']
    for r in paired:
        report.append(f'| {r["group"]} | {r["delta_mae"]:+.4f} | [{r["lower_95"]:+.4f}, {r["upper_95"]:+.4f}] |')
    report+=['','## 互斥的實際得分區間','','| 區間 | 筆數 | 模型MAE | 基準MAE | 平均殘差 |','| --- | ---: | ---: | ---: | ---: |']
    for b in actual_bands:
        report.append(f'| {BAND_LABELS[BANDS.index(b["band"])]} | {b["n"]} | {b["mae"]:.3f} | {b["baseline_mae"]:.3f} | {b["mean_residual"]:+.3f} |')
    report+=['','按預測得分分組的完整指標亦保存在 score_band_metrics.csv；空組留白，不假填零誤差。真值也參與殘差公式，不能只依真值分組就推論偏差原因。','',
        '## 事先凍結的模型比較','',
        '以下次要比較照協議預先列明，依固定模型／特徵順序呈現，不按測試排名改選部署模型。','',
        '| 模型版本 | 角色 | MAE | RMSE |','| --- | --- | ---: | ---: |']
    for r in scores:
        report.append(f'| {r["model_variant"]} | {r["role"]} | {r["mae"]:.4f} | {r["rmse"]:.4f} |')
    report+=['','## 延續驗證期案例球員','','這三人是在二月挑出的改善／退步／居中案例，不按測試表現另挑成功球員。','',
        '| 驗證期角色 | 球員 | 測試筆數 | 模型MAE | 基準MAE |','| --- | --- | ---: | ---: | ---: |']
    for r in case_results:
        model_mae=f'{r["mae"]:.3f}' if r['n'] else '無樣本'
        baseline_mae=f'{r["baseline_mae"]:.3f}' if r['n'] else '無樣本'
        report.append(f'| {r["role"]} | {r["player_id"]} | {r["n"]} | {model_mae} | {baseline_mae} |')
    report+=['','## 限制與交付','',
        '- 本次測試已使用，test_usage.json已記錄。後續不能用它調參再宣稱是未看過的最終測試；若開發新模型，需另找未使用資料驗證。',
        '- 主模型仍維持原權重及九項輸入；測試中任何次要模型的較好分數不會改寫預定選模結果。',
        '- 歷史先發來自賽後快照、季末資料可能有修訂，仍屬日期隔離的假設性回測；不能當成已證明的真實賽前服務成效。',
        '- 未評估預測出賽機率，也沒有先發缺席因果分析；只評估符合原規則且實際有上場的球員得分。',
        '- 高分極端案例存於largest_high_score_errors.csv，是事後診斷，不推定傷病、上場分鐘或缺席原因。',
        '- 未來展示應同時呈現整體與高分區間限制，並清楚標示資料不足及以出賽為前提。','',
        '檔案包含預先凍結協議、輸入雜湊、十組逐筆預測、整體／區間／球員指標、成對差異、三張圖、evaluated_candidate.json與validation.json。原始模型、調參與驗證分析均未覆寫。','',
        '腳本：scripts/final_test_evaluation.py；已存在的完成目錄會拒絕覆寫，不自動重跑最終測試。','']
    (out/'README.md').write_text('\n'.join(report),encoding='utf-8')
    disk=read_csv(out/'predictions_test.csv')
    for key in predictions:
        reread=evaluation_rows(records,[float(r['pred_'+key]) for r in disk])
        assert summarize(reread)==summarize(all_rows[key])
    assert sum(r['n'] for r in bands if r['grouping']=='actual')==6979
    assert sum(p['n'] for p in players)==6979
    assert sha(out/'test_protocol.json')==protocol_sha
    assert all(sha(ROOT/p)==h for p,h in {**frozen_sources,**test_hashes}.items())
    dump(out/'validation.json',dict(verified=True,test_evaluated=True,test_consumed=True,n=6979,
        primary_model=primary,model_fits_performed=0,preprocessing_fits_performed=0,model_reselected=False,
        protocol_sha256=protocol_sha,source_files_unchanged=True,source_sha256={**frozen_sources,**test_hashes},
        exported_model_scores_verified=10,primary_metrics=main_stats,
        output_sha256={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file() and p.name!='validation.json'}))
    print(json.dumps(dict(verified=True,primary_model=primary,metrics=main_stats,high_score_groups=high,paired=paired),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
