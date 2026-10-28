"""Bounded exploratory comparison; never reads targets after February 2026."""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits

from scripts.build_features_baseline import read_csv,save_csv,sha
from scripts.compare_models import BASIC,metrics,validate_development_rows
from scripts.recent_game_features import canonical_date,number
from scripts.recent_change_features import NEW_FEATURES,change_features_for_request
from scripts.time_validation_tuning import build_folds,FitBudget,stratified_interval,dump,FOLDS

ROOT=Path(__file__).resolve().parents[1]
BASE=list(BASIC)+['is_starter']
VARIANTS={'v1':BASE,'v2':BASE+NEW_FEATURES}


def development_history(raw_rows):
    result=[]; seen=set()
    for row in raw_rows:
        day=canonical_date(row['GAMEDATE']).isoformat()
        # Date filtering precedes access to any held-out outcome value.
        if day>'2026-02-28':
            continue
        key=row['PLAYER'],row['MATCHUP']+'|'+day
        if key in seen:
            raise ValueError('Duplicated source game')
        seen.add(key)
        result.append(dict(player_id=key[0],game_id=key[1],GAMEDATE=day,team_id=row['TEAM'],
                           **{c:row[c] for c in ['PTS','MIN','REB','AST']}))
    return result


def augment_development_rows(samples,history):
    by_player=defaultdict(list); source={}; seen=set(); result=[]
    for row in history:
        by_player[row['player_id']].append(row)
        key=row['player_id'],row['game_id']
        if key in source:
            raise ValueError('Duplicated history source')
        source[key]=row
    for sample in samples:
        day=canonical_date(sample['GAMEDATE']).isoformat(); split=sample['split']
        if not ((split=='train' and '2025-10-21'<=day<='2026-01-31') or
                (split=='validation' and '2026-02-01'<=day<='2026-02-28')):
            raise ValueError('Only original development dates are allowed')
        key=sample['player_id'],sample['game_id']
        if key in seen or key not in source:
            raise ValueError('Duplicate sample or missing target source')
        seen.add(key)
        target=source[key]
        if target['GAMEDATE']!=day or number(target['MIN']) in (None,0) or number(target['PTS'],integer=True)!=float(sample['target_pts']):
            raise ValueError('Target eligibility/label differs from original cohort')
        feature=change_features_for_request(by_player[key[0]],sample)
        if not feature['eligible'] or any(not np.isclose(float(sample[c]),feature[c],rtol=0,atol=1e-12) for c in BASIC):
            raise ValueError('Original feature/window mismatch; no silent exclusion')
        result.append(dict(sample,**{c:feature[c] for c in NEW_FEATURES},
                           **{c:feature[c] for c in ['history_count','last_history_date','history_game_ids_3','history_game_ids_5','history_game_ids_10']}))
    return result


def group_scores(actual,predicted,baseline):
    y,p,b=[np.asarray(v,dtype=float) for v in (actual,predicted,baseline)]
    if y.ndim!=1 or y.shape!=p.shape or y.shape!=b.shape or not np.isfinite(np.concatenate([y,p,b])).all():
        raise ValueError('Unaligned or nonfinite group input')
    result=[]
    for name,mask in [('all',np.ones(len(y),dtype=bool)),('20+',y>=20),('30+',y>=30)]:
        score=metrics(y[mask],p[mask]) if mask.any() else dict(n=0,mae=None,rmse=None)
        base=metrics(y[mask],b[mask]) if mask.any() else dict(mae=None,rmse=None)
        result.append(dict(group=name,**score,baseline_mae=base['mae'],baseline_rmse=base['rmse'],
                           mean_residual=float((y[mask]-p[mask]).mean()) if mask.any() else None))
    return result


def matrix(rows,columns):
    result=np.asarray([[float(r[c]) for c in columns] for r in rows])
    if not len(rows) or not np.isfinite(result).all():
        raise ValueError('Empty/nonfinite model features')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/week2/recent_change_v2')
    args=parser.parse_args(); out=args.out
    if out.exists():
        raise ValueError('Output already exists; do not overwrite an experiment')
    old=ROOT/'outputs/week2/time_validation_tuning'
    inputs=[ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv',
            ROOT/'outputs/week1/team_lineup_features/train.csv',ROOT/'outputs/week1/team_lineup_features/validation.csv',
            old/'common_parameters.json',old/'predictions_time_validation.csv',old/'predictions_february.csv']
    protected=list((old/'models').glob('*'))+[ROOT/'streamlit_app.py',ROOT/'nba_dashboard/data.py',
                  ROOT/'nba_dashboard/views.py',ROOT/'outputs/week2/final_test/evaluated_candidate.json',
                  ROOT/'requirements.txt',ROOT/'requirements-modeling.txt']
    before={str(p.relative_to(ROOT)):sha(p) for p in inputs+protected}
    params=json.loads(inputs[3].read_text(encoding='utf-8'))['selected']['xgboost']['params']
    original_train,original_val=read_csv(inputs[1]),read_csv(inputs[2])
    validate_development_rows(original_train,original_val)
    history=development_history(read_csv(inputs[0]))
    train=augment_development_rows(original_train,history)
    val=augment_development_rows(original_val,history)
    train.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    val.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    if (len(train),len(val))!=(13281,3462):
        raise ValueError('Original development cohort changed')
    folds=build_folds(train)
    out.mkdir(parents=True); (out/'models').mkdir(); (out/'features').mkdir()
    # Write the fixed protocol before the first fit. No adaptive search follows.
    dump(out/'experiment_config.json',dict(variants=VARIANTS,parameters=params,fit_cap=8,
         folds=[dict(name=f[0],train_end=f[1],validation_start=f[2],validation_end=f[3]) for f in FOLDS],
         exploratory_after_test_observed=True,selection_data_reused=True,test_scoring_allowed=False,
         auto_promotion=False,bootstrap_iterations=2000,seed=42))
    for name,rows in [('train',train),('validation',val)]:
        save_csv(out/'features'/f'{name}.csv',rows,list(rows[0]))
    definitions=[('pts_last_1','上一場得分'),('min_mean_3','前3場平均上場分鐘'),
                 ('min_change_last_vs_5','上一場分鐘減前5場平均分鐘'),('pts_std_5','前5場得分標準差（母體，ddof=0）'),
                 ('min_std_5','前5場分鐘標準差（母體，ddof=0）')]
    save_csv(out/'feature_dictionary.csv',[dict(feature=c,description=d,rule='同球季、嚴格前日、MIN>0；不足五場不預測') for c,d in definitions],['feature','description','rule'])
    controls={}
    for period,path in [('CV',inputs[4]),('February',inputs[5])]:
        controls[period]={(r.get('fold','February'),r['player_id'],r['game_id']):r for r in read_csv(path)}
    scores=[]; fit_log=[]; predictions={}; budget=FitBudget(8); reload_ok=True
    with threadpool_limits(limits=2):
        for period,tr,va in folds+[('February',train,val)]:
            y=np.asarray([float(r['target_pts']) for r in va]); base=np.asarray([float(r['pts_mean_5']) for r in va])
            wide=[{c:r[c] for c in ['player_id','game_id','GAMEDATE','target_pts']}|dict(period=period,pred_baseline=float(b)) for r,b in zip(va,base)]
            for variant,columns in VARIANTS.items():
                xtr,xva=matrix(tr,columns),matrix(va,columns)
                model=XGBRegressor(**params); budget.consume()
                model.fit(xtr,np.asarray([float(r['target_pts']) for r in tr]))
                pred=model.predict(xva).astype(float)
                if not np.isfinite(pred).all():
                    raise ValueError('Nonfinite predictions')
                modelpath=out/'models'/f'{period}_{variant}.joblib'
                joblib.dump(dict(model=model,columns=columns,params=params,research_only=True,
                                trained_through=max(r['GAMEDATE'] for r in tr)),modelpath)
                reloaded=joblib.load(modelpath)
                reload_ok &= np.array_equal(pred,reloaded['model'].predict(matrix(va,reloaded['columns'])).astype(float))
                fit_log.append(dict(period=period,variant=variant,n_train=len(tr),n_validation=len(va),fit_number=budget.count))
                for r,p in zip(wide,pred):
                    r['pred_'+variant]=float(p)
                scores.extend(dict(period=period,variant=variant,**s) for s in group_scores(y,pred,base))
            for r in wide:
                is_feb=period=='February'; control=controls['February' if is_feb else 'CV'][(period,r['player_id'],r['game_id'])]
                col='pred_xgboost_starter' if is_feb else 'pred_xgboost_02_starter'
                if abs(r['pred_v1']-float(control[col]))>1e-12 or float(r['target_pts'])!=float(control['target_pts']):
                    raise ValueError('Rebuilt v1 differs from frozen control')
            predictions[period]=wide
    cv=sum([predictions[name] for name,_,_ in folds],[])
    for variant in VARIANTS:
        scores.extend(dict(period='CV_pooled',variant=variant,**s) for s in group_scores(
            [r['target_pts'] for r in cv],[r['pred_'+variant] for r in cv],[r['pred_baseline'] for r in cv]))
    intervals=[]
    for period,rows in [('CV_pooled',cv),('February',predictions['February'])]:
        interval=stratified_interval([r['target_pts'] for r in rows],[r['pred_v1'] for r in rows],[r['pred_v2'] for r in rows],
                                    [r['GAMEDATE'] for r in rows],[r['period'] for r in rows])
        intervals.append(dict(period=period,**interval))
        save_csv(out/f'predictions_{period}.csv',rows,list(rows[0]))
    save_csv(out/'scores.csv',scores,list(scores[0])); save_csv(out/'fit_log.csv',fit_log,list(fit_log[0]))
    dump(out/'paired_intervals.json',intervals)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,period in zip(axes,['CV_pooled','February']):
        records=[next(s for s in scores if s['period']==period and s['variant']==v and s['group']=='all') for v in VARIANTS]
        ax.bar(['9 features','14 features'],[s['mae'] for s in records]); ax.set(title=period,ylabel='MAE (points)')
        for i,s in enumerate(records):
            ax.text(i,s['mae'],f"{s['mae']:.4f}",ha='center',va='bottom')
        ax.set_ylim(0,max(s['mae'] for s in records)*1.15)
    fig.tight_layout(); fig.savefig(out/'mae_comparison.png',dpi=160); plt.close(fig)
    lines=['# 近期變化特徵研究比較','',
           '新增上一場得分、前3場平均分鐘、上一場分鐘相對前5場變化、前5場得分與分鐘標準差，共14項特徵。',
           '使用原XGBoost設定，固定8次訓練；原九項模型、Streamlit及最終測試成果均保留。','',
           '|期間|樣本|九項 MAE|十四項 MAE|九項 RMSE|十四項 RMSE|','|---|---:|---:|---:|---:|---:|']
    for period in ['F1','F2','F3','CV_pooled','February']:
        a,b=[next(s for s in scores if s['period']==period and s['variant']==v and s['group']=='all') for v in VARIANTS]
        lines.append(f"|{period}|{a['n']}|{a['mae']:.4f}|{b['mae']:.4f}|{a['rmse']:.4f}|{b['rmse']:.4f}|")
    lines+=['','誤差差異定義為「十四項減九項」，負數代表改善。日期群組重抽2000次，CV依輪次分層：']
    for entry in intervals:
        lines.append(f"- {entry['period']}：MAE差 {entry['delta_mae']:.4f}，95%區間 [{entry['lower_95']:.4f}, {entry['upper_95']:.4f}]。")
    lines+=['','20+與30+使用同一批驗證樣本依實際得分分組，細節見 scores.csv。殘差為實際減預測，正值代表低估。',
            '本次想法是在看過舊測試表現後提出，且重用既有開發資料，屬探索結果；不是獨立的新測試證據。',
            '不計算三月至四月的v2成績、不改選正式模型。需在尚未檢視的新球季資料另定協議驗證後，才考慮更換。',
            '歷史先發仍來自賽後名單；成果屬日期隔離的假設性回測。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if not reload_ok or budget.count!=8 or any(sha(ROOT/p)!=value for p,value in before.items()):
        raise ValueError('Reload, fit budget or protected source integrity failed')
    checks=dict(train_n=len(train),february_n=len(val),cv_n=len(cv),fits=budget.count,
                exact_v1_control_reproduced=True,model_reload_equal=bool(reload_ok),same_cohort=True,
                test_metrics_computed=False,production_promoted=False,source_sha256=before,
                output_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    dump(out/'validation.json',checks)
    print(json.dumps(dict(output=str(out),fits=budget.count,intervals=intervals),ensure_ascii=False))


if __name__=='__main__':
    main()
