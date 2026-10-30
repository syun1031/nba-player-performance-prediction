"""Twelve fixed fits on a common opponent-eligible development cohort."""
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

from scripts.build_features_baseline import read_csv,save_csv,sha
from scripts.compare_models import validate_development_rows
from scripts.experiment_recent_changes import ROOT,BASE,matrix,group_scores
from scripts.recent_change_features import NEW_FEATURES
from scripts.opponent_context import build_team_games,opponent_features,CONTEXT_FEATURES
from scripts.team_lineup_features import TeamHistoryIndex
from scripts.recent_game_features import canonical_date
from scripts.merge_game_rosters import roster_index
from scripts.time_validation_tuning import build_folds,FitBudget,stratified_interval,dump,FOLDS
from scripts.experiment_two_stage import control_predictions,save_rows
from nba_dashboard.data import expected_hash

VARIANTS={'common9':BASE,'common14':BASE+NEW_FEATURES,'opponent16':BASE+NEW_FEATURES+CONTEXT_FEATURES}
ALL_VARIANTS=['baseline','frozen9','frozen14',*VARIANTS]


def augment_opponent_rows(rows,team_games):
    index=team_games if isinstance(team_games,TeamHistoryIndex) else TeamHistoryIndex(team_games)
    result=[]; excluded=[]; seen=set()
    for row in rows:
        day=canonical_date(row['GAMEDATE']).isoformat(); split=row['split']; key=row['player_id'],row['game_id']
        if not ((split=='train' and '2025-10-21'<=day<='2026-01-31') or
                (split=='validation' and '2026-02-01'<=day<='2026-02-28')) or key in seen:
            raise ValueError('Duplicate or out-of-development sample')
        seen.add(key); context=opponent_features(index,row); merged=dict(row,**context)
        (result if context['opponent_eligible'] else excluded).append(merged)
    return result,excluded


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/week2/opponent_context_v4')
    out=parser.parse_args().out
    if out.exists():
        raise ValueError('Output exists; refuse to overwrite')
    recent=ROOT/'outputs/week2/recent_change_v2'; old=ROOT/'outputs/week2/time_validation_tuning'
    raw=ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv'
    roster_path=raw.with_name('nba_2025_26_regular_season_game_rosters.csv')
    originals=[ROOT/f'outputs/week1/team_lineup_features/{n}.csv' for n in ['train','validation']]
    inputs=[raw,roster_path,*originals,recent/'features/train.csv',recent/'features/validation.csv',
            recent/'predictions_CV_pooled.csv',recent/'predictions_February.csv',recent/'validation.json',
            old/'common_parameters.json',ROOT/'outputs/week1/features_baseline/validation.json']
    protected=[p for folder in [old,recent,ROOT/'outputs/week2/two_stage_v3',ROOT/'outputs/week2/final_test']
               for p in folder.rglob('*') if p.is_file()]
    protected += [ROOT/'streamlit_app.py',ROOT/'nba_dashboard/data.py',ROOT/'nba_dashboard/views.py',
                  ROOT/'requirements.txt',ROOT/'requirements-modeling.txt']
    before={str(p.relative_to(ROOT)):sha(p) for p in set(inputs+protected)}
    meta=json.loads((recent/'validation.json').read_text(encoding='utf-8'))
    for p in inputs[4:8]:
        if sha(p)!=expected_hash(meta['output_sha256'],p.relative_to(recent)):
            raise ValueError('V2 feature/control version changed')
    raw_meta=json.loads(inputs[-1].read_text(encoding='utf-8'))
    if sha(raw)!=expected_hash(raw_meta['source_sha256'],raw.relative_to(ROOT)):
        raise ValueError('Original raw source changed')
    rosters=roster_index([r for r in read_csv(roster_path) if r['GAMEDATE']<='2026-02-28'])
    teams=build_team_games(read_csv(raw),rosters)
    original_train,original_val=[read_csv(p) for p in originals]; validate_development_rows(original_train,original_val)
    bases=[read_csv(recent/f'features/{n}.csv') for n in ['train','validation']]
    for rows,original in zip(bases,[original_train,original_val]):
        index={(r['player_id'],r['game_id']):r for r in rows}
        if len(index)!=len(rows) or len(rows)!=len(original) or any(
                any(index.get((r['player_id'],r['game_id']),{}).get(k)!=v for k,v in r.items()) for r in original):
            raise ValueError('V2 base cohort/original fields differ')
    train,excluded=augment_opponent_rows(bases[0],teams); val,excluded_val=augment_opponent_rows(bases[1],teams)
    if (len(teams),len(train),len(excluded),len(val),len(excluded_val))!=(1790,13221,60,3462,0):
        raise ValueError('Unexpected team-game or common-cohort coverage')
    for rows in [train,val]:
        rows.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    folds=build_folds(train)
    if [(len(tr),len(va)) for _,tr,va in folds]!=[(4198,4104),(8302,2422),(10724,2497)]:
        raise ValueError('Unexpected fixed fold sizes')
    params=json.loads((old/'common_parameters.json').read_text(encoding='utf-8'))['selected']['xgboost']['params']
    cv_control=read_csv(recent/'predictions_CV_pooled.csv'); feb_control=read_csv(recent/'predictions_February.csv')
    out.mkdir(parents=True); (out/'models').mkdir(); (out/'features').mkdir()
    # This protocol is persisted before the first model fit.
    dump(out/'experiment_config.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        params=params,variants=VARIANTS,fit_cap=12,folds=FOLDS,common_train_n=13221,excluded_train_n=60,
        common_training_required_for_all_models=True,eval_cohort_unchanged=True,
        opponent_min_history=5,source='full player logs checked against rosters',true_pace_available=False,
        overtime_adjusted=False,test_scoring_allowed=False,auto_promotion=False,
        exploratory_after_test_observed=True,development_reused=True,bootstrap_iterations=2000,seed=42))
    save_rows(out/'team_games.csv',teams); save_rows(out/'excluded.csv',excluded)
    save_rows(out/'features/train.csv',train); save_rows(out/'features/validation.csv',val)
    save_rows(out/'feature_dictionary.csv',[
        dict(feature=CONTEXT_FEATURES[0],description='對手前五場平均失分',limitation='非每百回合防守效率；含延長賽'),
        dict(feature=CONTEXT_FEATURES[1],description='對手前五場雙方總分平均',limitation='得分環境代理，不是回合數或Pace')])
    budget=FitBudget(12); fit_log=[]; predictions={}; reloaded=True
    with threadpool_limits(limits=2):
        for period,tr,va in folds+[('February',train,val)]:
            controls=[r for r in cv_control if r['period']==period] if period!='February' else feb_control
            frozen9=control_predictions(va,controls,'pred_v1'); frozen14=control_predictions(va,controls,'pred_v2')
            labels=np.asarray([float(r['target_pts']) for r in tr])
            rows=[{k:r[k] for k in ['player_id','game_id','GAMEDATE','target_pts','opponent_id','opponent_history_count',
                                   'opponent_last_history_date','opponent_history_game_ids_5',*CONTEXT_FEATURES]}|
                  dict(period=period,pred_baseline=float(r['pts_mean_5']),pred_frozen9=float(frozen9[i]),pred_frozen14=float(frozen14[i]))
                  for i,r in enumerate(va)]
            for variant,columns in VARIANTS.items():
                model=XGBRegressor(**params); xtr=matrix(tr,columns); xva=matrix(va,columns)
                budget.consume(); model.fit(xtr,labels); pred=model.predict(xva).astype(float)
                no_labels=[{k:v for k,v in r.items() if not k.startswith('target_')} for r in va]
                if not np.isfinite(pred).all() or not np.array_equal(pred,model.predict(matrix(no_labels,columns)).astype(float)):
                    raise ValueError('Nonfinite or outcome-dependent prediction')
                modelpath=out/'models'/f'{period}_{variant}.joblib'
                joblib.dump(dict(model=model,columns=columns,params=params,research_only=True,n_train=len(tr),
                                 trained_through=max(r['GAMEDATE'] for r in tr)),modelpath)
                loaded=joblib.load(modelpath)
                reloaded &= np.array_equal(pred,loaded['model'].predict(matrix(no_labels,loaded['columns'])).astype(float))
                for r,p in zip(rows,pred):
                    r['pred_'+variant]=float(p)
                fit_log.append(dict(period=period,variant=variant,fit_number=budget.count,n_train=len(tr),n_validation=len(va),
                                    train_end=max(r['GAMEDATE'] for r in tr),validation_start=min(r['GAMEDATE'] for r in va)))
            predictions[period]=rows
    cv=sum([predictions[name] for name,_,_ in folds],[]); predictions['CV_pooled']=cv
    if len(cv)!=9023 or len({(r['player_id'],r['game_id']) for r in cv})!=9023:
        raise ValueError('Wrong pooled validation cohort')
    scores=[]; intervals=[]
    for period,rows in predictions.items():
        for variant in ALL_VARIANTS:
            scores.extend(dict(period=period,variant=variant,**s) for s in group_scores(
                [r['target_pts'] for r in rows],[r['pred_'+variant] for r in rows],[r['pred_baseline'] for r in rows]))
        if period in ['CV_pooled','February']:
            save_rows(out/f'predictions_{period}.csv',rows)
            for reference in ['common9','common14','frozen9']:
                diff=stratified_interval([r['target_pts'] for r in rows],[r['pred_'+reference] for r in rows],
                    [r['pred_opponent16'] for r in rows],[r['GAMEDATE'] for r in rows],[r['period'] for r in rows])
                intervals.append(dict(period=period,reference=reference,**diff))
    save_rows(out/'scores.csv',scores); save_rows(out/'fit_log.csv',fit_log); dump(out/'paired_intervals.json',intervals)
    make_report(out,scores,intervals)
    if budget.count!=12 or not reloaded or any(sha(ROOT/p)!=h for p,h in before.items()):
        raise ValueError('Fit cap/reload/protected source check failed')
    code=[Path(__file__),ROOT/'scripts/opponent_context.py',ROOT/'scripts/experiment_two_stage.py',
          ROOT/'scripts/time_validation_tuning.py']
    dump(out/'validation.json',dict(fits=12,team_game_n=len(teams),physical_game_n=len(teams)//2,
         train_n=len(train),excluded_n=len(excluded),february_n=len(val),cv_n=len(cv),common_train_cohort=True,
         eval_cohort_unchanged=True,controls_exact=True,roster_coverage_verified=True,
         model_reload_equal=bool(reloaded),test_metrics_computed=False,production_promoted=False,
         source_sha256=before,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},
         environment=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in
                          ['numpy','scikit-learn','xgboost','joblib','matplotlib']}),
         output_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(output=str(out),fits=12,intervals=intervals),ensure_ascii=False))


def make_report(out,scores,intervals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=dict(baseline='五場均值',frozen9='原凍結九項',frozen14='原凍結十四項',
               common9='共同訓練九項',common14='共同訓練十四項',opponent16='加入對手十六項')
    lines=['# 對手失分與得分環境研究','',
           '新增對手前五場平均失分與雙方總得分平均。只使用同球季嚴格前日的完整比賽，不含當場或未來結果。',
           '原始完整球員分數合計並以名單核對，1790筆球隊視角對應895場實際比賽；中立場不依主客方向重複計算。',
           '缺少出手、罰球、進攻籃板、失誤及精確延長賽時間，因此不是正式Pace或每百回合防守效率，且包含延長賽得分。',
           '60筆早期訓練的對手不足五場，共同九項／十四項／十六項一起排除。三輪與二月驗證樣本完全不變。',
           '共12次固定參數fit，沒有追加調參；共同訓練模型間才是特徵效果的公平對照。原凍結模型的訓練仍為13281筆，另列作線上參考。','',
           '|期間|樣本|版本|MAE|RMSE|','|---|---:|---|---:|---:|']
    for period in ['F1','F2','F3','CV_pooled','February']:
        for variant in ALL_VARIANTS:
            s=next(r for r in scores if r['period']==period and r['variant']==variant and r['group']=='all')
            lines.append(f"|{period}|{s['n']}|{names[variant]}|{s['mae']:.4f}|{s['rmse']:.4f}|")
    lines+=['','## 高得分場次','', '|期間|得分組|樣本|版本|MAE|RMSE|','|---|---|---:|---|---:|---:|']
    for r in scores:
        if r['period'] in ['CV_pooled','February'] and r['group']!='all':
            lines.append(f"|{r['period']}|{r['group']}|{r['n']}|{names[r['variant']]}|{r['mae']:.4f}|{r['rmse']:.4f}|")
    lines+=['','## 日期群組成對差異','', '差異為十六項減對照，負值代表改善。2000次日期群組重抽、seed42；CV在各輪內分層。']
    for r in intervals:
        lines.append(f"- {r['period']}／{names[r['reference']]}：MAE差 {r['delta_mae']:.4f}，95%區間 [{r['lower_95']:.4f}, {r['upper_95']:.4f}]。")
    lines+=['','## 判讀與限制','',
            '這是看過舊測試後提出、重用開發資料的探索；不能當作未見資料的驗證。三月至四月不產生新預測，正式模型與Streamlit不替換。',
            '平均失分同時受對手強弱與比賽環境影響，只能作背景特徵，不能單獨解釋成對手防守造成的因果效果。',
            '得分環境不是實際回合數；正式Pace須補齊可靠的歷史球隊比賽統計。歷史先發仍為賽後快照，屬日期隔離的假設性回測。',
            '建議比較共同訓練十四項與十六項的整體MAE／RMSE、高分表現與各輪一致性；任何正式升級都需尚未檢視的新資料另定驗證。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,period in zip(axes,['CV_pooled','February']):
        variants=['frozen9','common9','common14','opponent16']
        values=[next(r['mae'] for r in scores if r['period']==period and r['variant']==v and r['group']=='all') for v in variants]
        ax.bar(['Frozen9','Common9','Common14','Opponent16'],values)
        for i,v in enumerate(values): ax.text(i,v,f'{v:.4f}',ha='center',va='bottom')
        ax.set(title=period,ylabel='Points MAE',ylim=(0,max(values)*1.15))
    fig.tight_layout(); fig.savefig(out/'mae_comparison.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    main()
