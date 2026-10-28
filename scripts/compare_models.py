"""Fixed-setting development comparison; never read or evaluate test rows."""
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path
import importlib.metadata

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

if __package__:
    from .build_features_baseline import read_csv, save_csv, sha
else:
    from build_features_baseline import read_csv, save_csv, sha

ROOT = Path(__file__).resolve().parents[1]
BASIC = ['pts_mean_5','pts_mean_10','min_mean_5','reb_mean_5','ast_mean_5',
         'is_home','is_neutral_site','rest_days']
SETTINGS = {
    'linear': {'fit_intercept': True},
    'tree': {'max_depth': 6, 'min_samples_leaf': 30, 'random_state': 42},
    'xgboost': {'n_estimators': 300, 'max_depth': 3, 'learning_rate': .05,
                'min_child_weight': 1, 'subsample': 1., 'colsample_bytree': 1.,
                'reg_alpha': 0., 'reg_lambda': 1., 'objective': 'reg:squarederror',
                'tree_method': 'hist', 'device': 'cpu', 'n_jobs': 2, 'random_state': 42},
}


def validate_development_rows(train, validation):
    seen = set()
    for split, rows, first, last in [('train',train,'2025-10-21','2026-01-31'),
                                     ('validation',validation,'2026-02-01','2026-02-28')]:
        if not rows:
            raise ValueError('Empty development split')
        for row in rows:
            day = row['GAMEDATE']
            if date.fromisoformat(day).isoformat() != day or not first <= day <= last or row['split'] != split:
                raise ValueError('Invalid development split or date; test rows are forbidden')
            key = row['player_id'], row['game_id']
            if key in seen:
                raise ValueError('Duplicate or overlapping sample key')
            seen.add(key)


def feature_groups(columns):
    if len(set(columns)) != len(columns) or 'is_starter' not in columns:
        raise ValueError('Invalid feature definition')
    for column in columns:
        if column not in BASIC + ['is_starter','teammate_unknown_count'] and not re.fullmatch(r'teammate_[0-9]{4}',column):
            raise ValueError(f'Forbidden model input: {column}')
    basic = [c for c in columns if c in BASIC]
    return dict(basic=basic, starter=basic+['is_starter'], teammates=list(columns))


def paired_mae_interval(actual, old, new, dates, iterations=2000, seed=42):
    actual,old,new = [np.asarray(v,dtype=float) for v in (actual,old,new)]
    if not (actual.ndim == old.ndim == new.ndim == 1 and len(actual)==len(old)==len(new)==len(dates)):
        raise ValueError('Unaligned paired samples')
    if not np.isfinite(np.concatenate([actual,old,new])).all() or iterations < 1:
        raise ValueError('Invalid paired values')
    groups, index = np.unique(dates,return_inverse=True)
    if len(groups)<2:
        raise ValueError('At least two dates are required')
    delta = np.abs(new-actual)-np.abs(old-actual)
    sums = np.bincount(index,weights=delta)
    counts = np.bincount(index)
    weights = np.random.default_rng(seed).multinomial(len(groups),[1/len(groups)]*len(groups),size=iterations)
    samples = (weights@sums)/(weights@counts)
    lo,hi = np.quantile(samples,[.025,.975])
    return dict(delta_mae=float(delta.mean()),lower_95=float(lo),upper_95=float(hi),
                date_groups=len(groups),iterations=iterations,seed=seed)


def metrics(actual, predicted):
    if len(actual)==0 or len(actual)!=len(predicted) or not np.isfinite(predicted).all():
        raise ValueError('Missing or nonfinite predictions')
    return dict(n=len(actual),mae=float(mean_absolute_error(actual,predicted)),
                rmse=float(np.sqrt(mean_squared_error(actual,predicted))))


def model_for(name):
    if name=='linear':
        return make_pipeline(StandardScaler(),LinearRegression(**SETTINGS[name]))
    if name=='tree':
        return DecisionTreeRegressor(**SETTINGS[name])
    if name=='xgboost':
        return XGBRegressor(**SETTINGS[name])
    raise ValueError(name)


def main():
    source = ROOT/'outputs/week1/teammate_encoding'
    out = ROOT/'outputs/week2/model_comparison'
    protected = [p for folder in ['features_baseline','team_lineup_features','teammate_encoding']
                 for p in (ROOT/'outputs/week1'/folder).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    spec = json.loads((source/'validation.json').read_text(encoding='utf-8'))
    if not spec['verified'] or spec['fit_split']!='train' or spec['fit_end_date']!='2026-01-31':
        raise ValueError('Unverified train-only encoding')
    if sha(source/'teammate_encoder.json') != spec['encoder_sha256']:
        raise ValueError('Encoder differs from verified version')
    groups = feature_groups(spec['numeric_feature_columns'])
    assert groups['basic']==BASIC and len(groups['starter'])==9 and len(groups['teammates'])==366
    rows = {s:read_csv(source/(s+'.csv')) for s in ['train','validation']}
    validate_development_rows(rows['train'],rows['validation'])
    assert (len(rows['train']),len(rows['validation']))==(13281,3462)
    # Independently reconcile the common cohort, eight baseline values and labels.
    for split in rows:
        original = read_csv(ROOT/'outputs/week1/features_baseline'/(split+'.csv'))
        reference = {(r['player_id'],r['game_id']):r for r in original}
        assert len(reference)==len(rows[split])
        for r in rows[split]:
            old = reference[(r['player_id'],r['game_id'])]
            assert all(float(r[c])==float(old[c]) for c in BASIC+['target_pts'])
            assert r['GAMEDATE']==old['GAMEDATE'] and r['pregame_verified']=='False'
    X = {s:{g:np.asarray([[float(r[c]) for c in columns] for r in data])
            for g,columns in groups.items()} for s,data in rows.items()}
    y = {s:np.asarray([float(r['target_pts']) for r in data]) for s,data in rows.items()}
    assert all(np.isfinite(a).all() for matrices in X.values() for a in matrices.values())
    assert all(np.isfinite(a).all() for a in y.values())
    assert all(float(r['teammate_unknown_count'])==0 for r in rows['train'])
    out.mkdir(parents=True,exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    versions = {name:importlib.metadata.version(name) for name in ['numpy','scikit-learn','xgboost','matplotlib','scipy','joblib']}
    config = dict(version='fixed-ablation-v1',python=sys.version,versions=versions,
                  settings=SETTINGS,feature_groups=groups,train_samples=len(y['train']),
                  validation_samples=len(y['validation']),scaling='Linear: StandardScaler fit on train only',
                  test_evaluated=False,hyperparameter_search=False,refit_with_validation=False,
                  source_sha256=hashes,bootstrap='Paired MAE difference; resample whole validation dates 2000 times, seed 42')
    (out/'experiment_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    prediction = {s:{'baseline':np.asarray([float(r['pts_mean_5']) for r in data])} for s,data in rows.items()}
    scores = []
    for split in rows:
        scores.append(dict(model='baseline',features='basic',split=split,**metrics(y[split],prediction[split]['baseline'])))
    with threadpool_limits(limits=2):
        for name in SETTINGS:
            for group,columns in groups.items():
                print(f'Training {name}/{group}: {len(columns)} inputs',flush=True)
                model = model_for(name).fit(X['train'][group],y['train'])
                if name=='linear':
                    assert int(model.steps[0][1].n_samples_seen_)==len(y['train'])
                key = name+'_'+group
                for split in rows:
                    prediction[split][key] = model.predict(X[split][group])
                    scores.append(dict(model=name,features=group,split=split,
                                       **metrics(y[split],prediction[split][key])))
                path = out/'models'/(key+'.joblib')
                joblib.dump(dict(model=model,feature_columns=columns,encoder_sha256=spec['encoder_sha256'],
                                 fit_split='train',fit_end_date='2026-01-31'),path,compress=3)
                restored = joblib.load(path)
                assert restored['feature_columns']==columns
                np.testing.assert_allclose(restored['model'].predict(X['validation'][group]),
                                           prediction['validation'][key],rtol=1e-12,atol=1e-12)
    save_csv(out/'metrics.csv',scores,['model','features','split','n','mae','rmse'])
    fields = ['player_id','game_id','GAMEDATE','split','target_pts','teammate_unknown_count']
    pred_fields = ['pred_'+key for key in prediction['train']]
    saved = []
    for split,data in rows.items():
        for i,row in enumerate(data):
            r = {f:row[f] for f in fields}
            r.update({'pred_'+k:float(v[i]) for k,v in prediction[split].items()})
            saved.append(r)
    save_csv(out/'predictions_development.csv',saved,fields+pred_fields)
    # Recompute each score from the exported predictions rather than trusting in-memory results.
    exported = read_csv(out/'predictions_development.csv')
    for score in scores:
        subset = [r for r in exported if r['split']==score['split']]
        key = 'baseline' if score['model']=='baseline' else score['model']+'_'+score['features']
        reread = metrics([float(r['target_pts']) for r in subset],np.asarray([float(r['pred_'+key]) for r in subset]))
        assert reread=={k:score[k] for k in ['n','mae','rmse']}
    paired = []
    for name in SETTINGS:
        for old,new in [('basic','starter'),('basic','teammates'),('starter','teammates')]:
            a,b = prediction['validation'][name+'_'+old],prediction['validation'][name+'_'+new]
            paired.append(dict(model=name,old_features=old,new_features=new,
                               **paired_mae_interval(y['validation'],a,b,[r['GAMEDATE'] for r in rows['validation']]),
                               delta_rmse=metrics(y['validation'],b)['rmse']-metrics(y['validation'],a)['rmse']))
    save_csv(out/'paired_differences.csv',paired,list(paired[0]))
    unknown = np.asarray([int(r['teammate_unknown_count'])>0 for r in rows['validation']])
    subgroup = []
    for group,mask in [('known_only',~unknown),('has_unknown',unknown)]:
        for key,p in prediction['validation'].items():
            subgroup.append(dict(group=group,model_variant=key,**metrics(y['validation'][mask],p[mask])))
    save_csv(out/'unknown_coverage_metrics.csv',subgroup,list(subgroup[0]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    val_scores = [r for r in scores if r['split']=='validation']
    fig,axes = plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,metric in zip(axes,['mae','rmse']):
        for j,group in enumerate(groups):
            values = [next(r[metric] for r in val_scores if r['model']==name and r['features']==group) for name in SETTINGS]
            bars = ax.bar(np.arange(3)+(j-1)*.25,values,.25,label=group)
            ax.bar_label(bars,fmt='%.3f',fontsize=8)
        ax.axhline(val_scores[0][metric],color='black',ls='--',label='Last-five mean')
        ax.set_xticks(np.arange(3),list(SETTINGS))
        ax.set_ylabel(metric.upper()+' (points; lower is better)')
        ax.set_title('February validation: 3,462 identical samples')
        ax.legend(fontsize=8)
    fig.savefig(out/'validation_comparison.png',dpi=180)
    plt.close(fig)
    winner = min(val_scores,key=lambda r:(r['mae'],r['rmse'],list(['baseline']+list(SETTINGS)).index(r['model'])))
    unchanged = all(sha(ROOT/path)==value for path,value in hashes.items())
    if not unchanged:
        raise ValueError('Source data changed during experiment')
    verification = dict(verified=True,source_files_unchanged=True,identical_cohort_and_targets_verified=True,
                        exported_metrics_verified=True,model_reload_predictions_verified=True,
                        scaler_fit_train_only=True,test_rows_read=False,test_metrics_computed=False,
                        n_train=len(y['train']),n_validation=len(y['validation']),
                        unknown_validation_samples=int(unknown.sum()),validation_date_groups=len(set(r['GAMEDATE'] for r in rows['validation'])),
                        best_validation_setting=winner,source_sha256=hashes,
                        output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['validation.json','README.md']})
    (out/'validation.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    names = dict(baseline='五場平均',linear='線性迴歸',tree='決策樹',xgboost='XGBoost')
    report = ['# 基礎與隊友模型比較','',
              '2026-10-22。固定訓練 13,281 筆、二月驗證 3,462 筆，三組特徵使用完全相同樣本與目標。測試集未讀取、未評分。',
              '驗證原有 3,579 筆正分鐘紀錄，其中 3,462 筆合格，覆蓋率 96.73%；三組特徵均涵蓋這些合格樣本的 100%，沒有因編碼未知姓名另刪樣本。',
              '',f"本次最低驗證 MAE：{names[winner['model']]}／{winner['features']}，MAE {winner['mae']:.4f}、RMSE {winner['rmse']:.4f} 分。這是開發期結果，尚非最終測試表現。",'',
              '## 三組輸入','',
              '- basic：八項基礎數值。',
              '- starter：基礎加自己是否先發，九項數值。',
              '- teammates：再加 356 位隊友旗標與未知人數，共 366 項。',
              '- 隊友旗標相對 starter 的差異，才是排除「自己是否先發」後的隊友編碼增益。','',
              '## 驗證結果','', '| 模型 | 特徵 | MAE | RMSE |','| --- | --- | ---: | ---: |']
    for r in val_scores:
        report.append(f"| {names[r['model']]} | {r['features']} | {r['mae']:.4f} | {r['rmse']:.4f} |")
    report += ['','## 成對比較','',
               '差值＝新組減舊組；負值表示誤差降低。百分比相對舊組 MAE。區間以整個驗證日期為群組重抽 2,000 次，保留同日樣本一起出現；僅作探索性不確定性參考，沒有處理跨日相依或多重比較。', '',
               '| 模型 | 新組相對舊組 | MAE 差值 | MAE 變化 | RMSE 差值 | MAE 差值 95% 區間 |',
               '| --- | --- | ---: | ---: | ---: | --- |']
    for r in paired:
        old = next(s for s in val_scores if s['model']==r['model'] and s['features']==r['old_features'])
        report.append(f"| {names[r['model']]} | {r['new_features']} vs {r['old_features']} | {r['delta_mae']:+.4f} | {r['delta_mae']/old['mae']*100:+.2f}% | {r['delta_rmse']:+.4f} | [{r['lower_95']:+.4f}, {r['upper_95']:+.4f}] |")
    own_effects = [r for r in paired if r['old_features']=='basic' and r['new_features']=='starter']
    identity_effects = [r for r in paired if r['old_features']=='starter']
    own_summary = '；'.join(f"{names[r['model']]} {r['delta_mae']:+.4f} 分" for r in own_effects)
    identity_summary = '；'.join(f"{names[r['model']]} {r['delta_mae']:+.4f} 分" for r in identity_effects)
    xgb_effect = next(r for r in identity_effects if r['model']=='xgboost')
    xgb_old = next(r for r in val_scores if r['model']=='xgboost' and r['features']=='starter')
    interval_note = ('區間跨過零，尚不足以確認有穩定改善' if xgb_effect['lower_95']<=0<=xgb_effect['upper_95']
                     else '區間未跨零，仍需其他時間區間及最終測試確認')
    subgroup_summary = []
    for group,label in [('known_only','皆已知隊友'),('has_unknown','含未知隊友')]:
        before = next(r['mae'] for r in subgroup if r['group']==group and r['model_variant']=='xgboost_starter')
        after = next(r['mae'] for r in subgroup if r['group']==group and r['model_variant']=='xgboost_teammates')
        subgroup_summary.append(f'{label}：{before:.3f} → {after:.3f}')
    report += ['','## 結論與建議','',
               '加入自己是否先發的 MAE 差異：'+own_summary+'。',
               '再加入隊友身分的 MAE 差異：'+identity_summary+'。差值為負才是改善。',
               f"XGBoost 的隊友身分 MAE 相對變化為 {xgb_effect['delta_mae']/xgb_old['mae']*100:+.2f}%；{interval_note}。完整陣容版本與八項基礎版本的差異包含自己是否先發的效果，不能全部歸給隊友身分。",
               'XGBoost 分組 MAE（加入隊友身分前 → 後）：'+'；'.join(subgroup_summary)+'。平均分數可能掩蓋不同覆蓋狀況。',
               '建議保留「自己是否先發」；隊友編碼先保留為研究候選，與較簡單的九項特徵版本一起做更多按時間驗證，再鎖定設定與最終測試，不直接認定更多特徵一定較好。','',
               '## 解讀與限制','',
               '- 只跑一組預先固定設定，沒有依驗證分數調參；參數與版本見 experiment_config.json。',
               '- 線性迴歸僅用訓練資料擬合標準化；樹模型不標準化。沒有使用球員 ID、日期、本場實際分鐘或得分作輸入；本場先發身分來自下述賽後快照。',
               '- multi-hot 與自己是否先發有線性相依（訓練隊友數為 5 減是否先發），線性模型可能受共線性及未見組合影響；不把個別係數解讀為因果。',
               '- 驗證 505 筆含未知隊友、2,957 筆皆已知，分組誤差見 unknown_coverage_metrics.csv；不可因分組結果回填驗證姓名。',
               '- 未知人數在訓練時皆為零，模型未學過未知隊友的效果，該欄主要供追蹤覆蓋。',
               '- 歷史先發來自賽後快照，缺少當時公布時間，因此只能視為日期隔離的假設性研究；不能把本結果宣稱為完整賽前重建。',
               '- 本次沒有「原本先發缺席」欄位，也不能將先發替換直接視為缺席或因果影響。',
               '- 後續依驗證集選擇設定並鎖定，再執行保留測試集一次最終評估。','',
               '## 檔案與重現','',
               '執行 `.venv/Scripts/python.exe scripts/compare_models.py` 可重建本目錄。需先按 requirements-modeling.txt 安裝套件。', '',
               '- metrics.csv：訓練／驗證 MAE、RMSE。',
               '- predictions_development.csv：各模型逐筆開發期預測。',
               '- paired_differences.csv：同一樣本的成對差異。',
               '- models/：模型與固定特徵順序；只載入本專案信任的模型檔。',
               '- experiment_config.json、validation.json：固定設定、來源雜湊與驗證結果。',
               '- validation_comparison.png：驗證誤差圖。', '',
               '原始資料、既有編碼與共同樣本清單均未修改；模型重載後預測及匯出分數重新計算核對通過。','']
    (out/'README.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(dict(verified=True,best=winner,validation=val_scores,paired=paired),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
