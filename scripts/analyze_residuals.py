"""Validation-only diagnostics for the locked candidate; no fitting or test scoring."""
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import joblib
from threadpoolctl import threadpool_limits

if __package__:
    from .build_features_baseline import read_csv,save_csv,sha
else:
    from build_features_baseline import read_csv,save_csv,sha

ROOT=Path(__file__).resolve().parents[1]
BANDS=['lt0','0_to_lt10','10_to_lt20','20_to_lt30','30_plus']
BAND_LABELS=['<0','0–<10','10–<20','20–<30','30+']
SUMMARY_FIELDS=['mae','rmse','mean_residual','median_absolute_error','p90_absolute_error',
                'within_5_rate','within_10_rate','actual_mean','prediction_mean','baseline_mae','baseline_rmse','delta_mae']


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def build_residuals(predictions,features,prediction_column):
    lookup={}
    for r in features:
        key=r['player_id'],r['game_id']
        if key in lookup:
            raise ValueError('Duplicate feature key')
        lookup[key]=r
    seen=set(); result=[]
    for r in predictions:
        key=r['player_id'],r['game_id']
        f=lookup.get(key)
        if key in seen or f is None or f['split']!='validation':
            raise ValueError('Missing, duplicated, or non-validation sample')
        day=r['GAMEDATE']
        if not '2026-02-01'<=day<='2026-02-28' or day!=f['GAMEDATE'] or not r['game_id'].endswith('|'+day):
            raise ValueError('Non-February or inconsistent sample date')
        actual,pred,baseline=[float(r[k]) for k in ['target_pts',prediction_column,'pred_baseline']]
        starter=float(f['is_starter']); recent=float(f['pts_mean_5'])
        unknown=float(r['teammate_unknown_count'])
        if (not all(math.isfinite(v) for v in [actual,pred,baseline,starter,recent,unknown]) or actual<0
                or actual!=float(f['target_pts']) or baseline!=recent or starter not in (0,1)
                or unknown<0 or not unknown.is_integer()):
            raise ValueError('Nonfinite, invalid, or inconsistent analysis values')
        result.append(dict(player_id=r['player_id'],game_id=r['game_id'],GAMEDATE=day,
            actual=actual,predicted=pred,baseline=baseline,residual=actual-pred,
            absolute_error=abs(actual-pred),baseline_residual=actual-baseline,
            baseline_absolute_error=abs(actual-baseline),is_starter=int(starter),teammate_unknown_count=int(unknown)))
        seen.add(key)
    if seen!=set(lookup) or not result:
        raise ValueError('Analysis must cover the complete common validation cohort')
    return result


def summarize(rows):
    if not rows:
        return dict(n=0,**dict.fromkeys(SUMMARY_FIELDS))
    a=np.array([r['actual'] for r in rows]); p=np.array([r['predicted'] for r in rows])
    b=np.array([r['baseline'] for r in rows]); e=a-p; absolute=np.abs(e)
    if not np.isfinite(np.concatenate([a,p,b])).all():
        raise ValueError('Nonfinite analysis values')
    mae=float(absolute.mean()); baseline_mae=float(np.abs(a-b).mean())
    return dict(n=len(rows),mae=mae,rmse=float(np.sqrt(np.mean(e*e))),mean_residual=float(e.mean()),
        median_absolute_error=float(np.median(absolute)),p90_absolute_error=float(np.quantile(absolute,.9)),
        within_5_rate=float(np.mean(absolute<=5)),within_10_rate=float(np.mean(absolute<=10)),
        actual_mean=float(a.mean()),prediction_mean=float(p.mean()),baseline_mae=baseline_mae,
        baseline_rmse=float(np.sqrt(np.mean((a-b)**2))),delta_mae=mae-baseline_mae)


def score_band(value):
    if not math.isfinite(value):
        raise ValueError('Nonfinite score')
    return BANDS[0 if value<0 else 1 if value<10 else 2 if value<20 else 3 if value<30 else 4]


def grouped_bands(rows,column):
    if column not in ['actual','predicted']:
        raise ValueError('Unknown grouping column')
    return [dict(grouping=column,band=band,**summarize([r for r in rows if score_band(r[column])==band])) for band in BANDS]


def player_profiles(rows):
    players=defaultdict(list)
    for r in rows:
        players[r['player_id']].append(r)
    return [dict(player_id=name,case_eligible=len(data)>=8,**summarize(data)) for name,data in sorted(players.items())]


def choose_cases(profiles):
    eligible=[p for p in profiles if p['n']>=8]
    if not eligible:
        raise ValueError('No player has at least eight validation samples for player-level cases')
    middle=float(np.median([p['mae'] for p in eligible]))
    rules=[('most_improved',lambda p:(p['delta_mae'],p['player_id'])),
           ('most_worsened',lambda p:(-p['delta_mae'],p['player_id'])),
           ('typical',lambda p:(abs(p['mae']-middle),p['player_id']))]
    return [dict(role=role,player_id=min(eligible,key=key)['player_id']) for role,key in rules]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/week2/residual_analysis')
    args=parser.parse_args(); out=args.output_dir
    if out.exists():
        raise FileExistsError('Use a new --output-dir; keep completed analyses intact')
    source=ROOT/'outputs/week2/time_validation_tuning'
    candidate_path=source/'selected_candidate.json'
    candidate=json.loads(candidate_path.read_text(encoding='utf-8'))
    prediction_path=source/candidate['validation_sample_manifest']
    feature_path=ROOT/'outputs/week1/team_lineup_features/validation.csv'
    model_path=source/candidate['model_file'] if candidate['model_file'] else None
    protected=[p for p in source.rglob('*') if p.is_file()]+[feature_path]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    if sha(prediction_path)!=candidate['validation_manifest_sha256']:
        raise ValueError('Locked predictions have changed')
    if sha(feature_path)!=candidate['source_sha256'][str(feature_path.relative_to(ROOT))]:
        raise ValueError('Validation features differ from the locked run')
    prediction_column='pred_'+('baseline' if candidate['result']['model']=='baseline'
                               else candidate['result']['model']+'_'+candidate['result']['features'])
    predictions=read_csv(prediction_path); features=read_csv(feature_path)
    rows=build_residuals(predictions,features,prediction_column)
    stats=summarize(rows)
    assert stats['n']==candidate['validation_samples']==3462
    assert math.isclose(stats['mae'],candidate['result']['mae'],rel_tol=1e-12)
    assert math.isclose(stats['rmse'],candidate['result']['rmse'],rel_tol=1e-12)
    if model_path:
        assert sha(model_path)==candidate['model_sha256']
        bundle=joblib.load(model_path)
        lookup={(r['player_id'],r['game_id']):r for r in features}
        X=np.array([[float(lookup[(r['player_id'],r['game_id'])][c]) for c in bundle['feature_columns']] for r in rows])
        with threadpool_limits(limits=2):
            restored=bundle['model'].predict(X)
        np.testing.assert_allclose(restored,[r['predicted'] for r in rows],rtol=1e-12,atol=1e-12)
    out.mkdir(parents=True)
    save_csv(out/'residuals_validation.csv',rows,list(rows[0]))
    bands=grouped_bands(rows,'actual')+grouped_bands(rows,'predicted')
    save_csv(out/'score_band_metrics.csv',bands,list(bands[0]))
    profiles=player_profiles(rows)
    save_csv(out/'player_metrics.csv',profiles,list(profiles[0]))
    cases=choose_cases(profiles)
    extremes=[dict(role='largest_underestimate',**max(rows,key=lambda r:(r['residual'],r['GAMEDATE'],r['player_id']))),
              dict(role='largest_overestimate',**min(rows,key=lambda r:(r['residual'],r['GAMEDATE'],r['player_id'])))]
    save_csv(out/'extreme_games.csv',extremes,list(extremes[0]))
    top=sorted(rows,key=lambda r:(-r['absolute_error'],r['GAMEDATE'],r['player_id']))[:20]
    save_csv(out/'largest_errors.csv',top,list(top[0]))
    profile_lookup={r['player_id']:r for r in profiles}
    case_summary=[dict(role=c['role'],**profile_lookup[c['player_id']]) for c in cases]
    save_csv(out/'case_selection.csv',case_summary,list(case_summary[0]))
    case_players=list(dict.fromkeys(c['player_id'] for c in cases+extremes))
    case_rows=[dict(r,case_roles='|'.join(c['role'] for c in cases+extremes if c['player_id']==r['player_id']))
               for r in rows if r['player_id'] in case_players]
    save_csv(out/'player_case_games.csv',case_rows,list(case_rows[0]))
    days=sorted(set(r['GAMEDATE'] for r in rows))
    daily=[dict(GAMEDATE=day,**summarize([r for r in rows if r['GAMEDATE']==day])) for day in days]
    save_csv(out/'daily_metrics.csv',daily,list(daily[0]))
    grouped=[]
    for column,values in [('is_starter',[0,1]),('has_unknown_teammate',[0,1])]:
        for value in values:
            subset=[r for r in rows if (r['is_starter'] if column=='is_starter' else int(r['teammate_unknown_count']>0))==value]
            grouped.append(dict(grouping=column,value=value,**summarize(subset)))
    save_csv(out/'lineup_group_metrics.csv',grouped,list(grouped[0]))
    dump(out/'summary.json',dict(candidate_model=candidate['result']['model'],candidate_features=candidate['result']['features'],
         residual_definition='actual_minus_prediction; positive=underestimated; negative=overestimated',
         metrics=stats,players=len(profiles),case_eligible_players=sum(p['case_eligible'] for p in profiles),
         case_min_validation_games=8,cases=cases,test_metrics_computed=False,model_fits_performed=0))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    actual=np.array([r['actual'] for r in rows]); predicted=np.array([r['predicted'] for r in rows])
    residual=actual-predicted
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    ax=axs[0,0]; ax.scatter(predicted,actual,s=9,alpha=.22,rasterized=True)
    end=max(actual.max(),predicted.max())+1
    ax.plot([0,end],[0,end],'k--',lw=1); ax.set(xlabel='Predicted points',ylabel='Actual points',title='Locked model: February validation (n=3,462)')
    ax=axs[0,1]; ax.scatter(predicted,residual,s=9,alpha=.22,rasterized=True)
    ax.axhline(0,color='black',ls='--'); ax.set(xlabel='Predicted points',ylabel='Actual minus predicted',title='Positive residual = underprediction')
    ax=axs[1,0]; ax.hist(residual,bins=40,alpha=.8,color='#237aab')
    ax.axvline(0,color='black',ls='--'); ax.set(xlabel='Residual (points)',ylabel='Samples',title=f'MAE {stats["mae"]:.3f}; mean residual {stats["mean_residual"]:+.3f}')
    ax=axs[1,1]; ax.plot(range(len(days)),[r['mean_residual'] for r in daily],marker='o',ms=3)
    ax.axhline(0,color='black',ls='--'); ticks=np.arange(0,len(days),3)
    ax.set_xticks(ticks,[days[i][5:] for i in ticks]); ax.set(xlabel='February game date',ylabel='Mean actual minus predicted',title='Daily signed error (different daily cohorts)')
    fig.savefig(out/'residual_diagnostics.png',dpi=170); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    actual_bands=[b for b in bands if b['grouping']=='actual' and b['n']]
    predicted_bands=[b for b in bands if b['grouping']=='predicted' and b['n']]
    for j,field in enumerate(['baseline_mae','mae']):
        bars=axs[0].bar(np.arange(len(actual_bands))+(j-.5)*.34,[b[field] for b in actual_bands],.34,label='Last-five mean' if j==0 else 'Locked model')
        axs[0].bar_label(bars,fmt='%.2f',fontsize=8)
    axs[0].set_xticks(range(len(actual_bands)),[BAND_LABELS[BANDS.index(b['band'])]+f'\nn={b["n"]}' for b in actual_bands])
    axs[0].set(xlabel='Actual score band (post-outcome grouping)',ylabel='MAE (points)',title='Errors by actual points'); axs[0].legend()
    bars=axs[1].bar(range(len(predicted_bands)),[b['mean_residual'] for b in predicted_bands],color='#237aab')
    axs[1].bar_label(bars,fmt='%+.2f',fontsize=9); axs[1].axhline(0,color='black',ls='--')
    axs[1].set_xticks(range(len(predicted_bands)),[BAND_LABELS[BANDS.index(b['band'])]+f'\nn={b["n"]}' for b in predicted_bands])
    axs[1].set(xlabel='Predicted score band',ylabel='Mean actual minus predicted',title='Signed error by predicted points')
    fig.savefig(out/'score_band_errors.png',dpi=170); plt.close(fig)
    fig,axs=plt.subplots(math.ceil(len(case_players)/2),2,figsize=(13,3.7*math.ceil(len(case_players)/2)),layout='constrained',squeeze=False)
    for ax,player in zip(axs.flat,case_players):
        games=sorted([r for r in rows if r['player_id']==player],key=lambda r:(r['GAMEDATE'],r['game_id']))
        x=range(len(games))
        for field,label,style in [('actual','Actual','o-'),('predicted','Locked model','s-'),('baseline','Last-five mean','--')]:
            ax.plot(x,[r[field] for r in games],style,label=label,ms=3)
        ax.set_xticks(x,[r['GAMEDATE'][5:] for r in games],rotation=45,fontsize=8)
        ax.set_title(f'{player}: n={len(games)}, MAE={profile_lookup[player]["mae"]:.2f}')
        ax.set_ylabel('Points'); ax.legend(fontsize=8)
    for ax in list(axs.flat)[len(case_players):]:
        ax.set_visible(False)
    fig.savefig(out/'player_cases.png',dpi=160); plt.close(fig)
    report=['# 二月驗證殘差分析與球員案例','',
        '2026-10-23。使用已鎖定的 XGBoost／starter 九項特徵版本，完整分析二月 3,462 筆驗證預測。沒有重新訓練、改參數、刪除大誤差樣本或評分測試集。','',
        '## 主要結果','',
        f'- MAE：{stats["mae"]:.4f} 分；RMSE：{stats["rmse"]:.4f} 分，與鎖定候選一致。',
        f'- 五場均值基準 MAE：{stats["baseline_mae"]:.4f}；模型平均絕對誤差相對下降 {-stats["delta_mae"]/stats["baseline_mae"]*100:.2f}%。這是二月開發結果，不是最終測試。',
        f'- 平均殘差：{stats["mean_residual"]:+.4f} 分；絕對誤差中位數 {stats["median_absolute_error"]:.3f}、第90百分位 {stats["p90_absolute_error"]:.3f} 分。',
        f'- 誤差不超過5分：{stats["within_5_rate"]:.1%}；不超過10分：{stats["within_10_rate"]:.1%}。不是預測區間，也不保證下一場覆蓋率。','',
        '殘差＝實際得分－預測得分；正值代表低估，負值代表高估。正負誤差會互相抵銷，因此平均殘差接近零不表示每場預測準確。','',
        '## 得分區間','',
        '| 實際得分區間 | 筆數 | 模型 MAE | 基準 MAE | 平均殘差 |','| --- | ---: | ---: | ---: | ---: |']
    for b in actual_bands:
        report.append(f'| {BAND_LABELS[BANDS.index(b["band"])]} | {b["n"]} | {b["mae"]:.3f} | {b["baseline_mae"]:.3f} | {b["mean_residual"]:+.3f} |')
    small=[b for b in predicted_bands if b['n']<20]
    small_note=('預測得分分組中，'+ '、'.join(BAND_LABELS[BANDS.index(b['band'])]+f' 只有 {b["n"]} 筆' for b in small)
                +'，樣本過少，只作描述，不據此推論校準或未來表現。') if small else ''
    report+=['','實際得分是賽後才知道的分組依據，而且參與殘差計算；不能單憑高分組低估，就判定某種原因或預測時能辨識高分場。另存預測得分區間及其平均殘差，便於分開觀察。空區間不填虛構的零誤差。',small_note,'',
             '## 球員案例：固定規則挑選','',
             f'全部 {len(profiles)} 位球員均保留統計；其中 {sum(p["case_eligible"] for p in profiles)} 位在二月至少有8筆合格驗證，才參與球員案例挑選。此門檻只為避免用一兩場概括球員，不影響模型的歷史五場資格。', '',
             '改善／退步以模型 MAE 減五場均值基準 MAE 排序；典型案例取球員 MAE 最接近合格球員中位數者。同分依姓名排序，角色可以重複對應同一人。不把這些事後挑選案例當作獨立驗證。','',
             '| 角色 | 球員 | 二月筆數 | 模型 MAE | 基準 MAE | MAE 差值 |','| --- | --- | ---: | ---: | ---: | ---: |']
    role_names={'most_improved':'改善最多','most_worsened':'退步最多','typical':'誤差居中'}
    for r in case_summary:
        report.append(f'| {role_names[r["role"]]} | {r["player_id"]} | {r["n"]} | {r["mae"]:.3f} | {r["baseline_mae"]:.3f} | {r["delta_mae"]:+.3f} |')
    report+=['','## 單場極端案例','']
    for r in extremes:
        label='低估最多' if r['residual']>0 else '高估最多'
        report.append(f'- {label}：{r["GAMEDATE"]}，{r["player_id"]}（{r["game_id"].split("|")[0]}），實際 {r["actual"]:.0f}、預測 {r["predicted"]:.2f}、殘差 {r["residual"]:+.2f} 分；前五場均值 {r["baseline"]:.2f}。')
    report+=['','這些例子顯示單場結果可能大幅偏離近期均值；目前資料不足以把差異歸因於傷病、本場分鐘、隊友缺席或其他原因，不倒填為賽前已知資訊。','',
        '## 使用限制與下一步','',
        '- 二月已用於選模，這份診斷只描述開發資料；不作獨立泛化證明。沒有因為看到案例而新增特徵、調參或改鎖定候選。',
        '- 球員與分組樣本量不同，極端案例及小樣本分組容易波動，不當作球員能力或模型對某人的永久排名。',
        '- 原模型未使用隊友編碼。未知隊友欄只作覆蓋分組，不表示它是目前模型輸入；分組差異也不代表未知身分造成誤差。',
        '- 歷史先發仍來自賽後快照，成果屬日期隔離的假設性回測，不是完整賽前重建。',
        '- 建議按既有鎖定設定進入預先列明的最終測試比較，完整呈現整體、區間與球員誤差，不再依本次診斷追加搜尋。','',
        '## 圖表與資料','',
        '- residual_diagnostics.png：實際與預測、殘差與預測、殘差分布、每日平均殘差。',
        '- score_band_errors.png：實際得分區間的模型／基準 MAE，以及預測得分區間平均殘差。',
        '- player_cases.png：固定規則案例球員的二月逐場實際、模型與五場均值。',
        '- residuals_validation.csv、score_band_metrics.csv、player_metrics.csv、case_selection.csv、player_case_games.csv、extreme_games.csv、largest_errors.csv、daily_metrics.csv、lineup_group_metrics.csv：完整分析數據。',
        '- summary.json、validation.json：數值摘要、來源雜湊及核對證據。','',
        '重現：.venv/Scripts/python.exe scripts/analyze_residuals.py。完成目錄不覆寫，需重跑時另指定 --output-dir。','']
    (out/'README.md').write_text('\n'.join(report),encoding='utf-8')
    disk=read_csv(out/'residuals_validation.csv')
    for original,saved in zip(rows,disk,strict=True):
        assert all(float(saved[k])==original[k] for k in ['actual','predicted','baseline','residual','absolute_error'])
    for column in ['actual','predicted']:
        assert sum(r['n'] for r in bands if r['grouping']==column)==3462
    assert sum(p['n'] for p in profiles)==3462
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    dump(out/'validation.json',dict(verified=True,n=3462,model_fits_performed=0,test_metrics_computed=False,
        model_reload_predictions_verified=True,source_files_unchanged=True,source_sha256=hashes,
        residuals_readback_verified=True,band_and_player_counts_reconciled=True,
        output_sha256={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file() and p.name!='validation.json'}))
    print(json.dumps(dict(verified=True,metrics=stats,cases=case_summary,extremes=extremes),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
