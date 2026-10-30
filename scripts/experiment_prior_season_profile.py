"""Eight fixed fits: prior-season profile on basic9, then combined20 with 2x."""
import json
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime,timezone
import joblib
import numpy as np
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
from scripts.build_features_baseline import read_csv,sha
from scripts.compare_models import validate_development_rows
from scripts.experiment_recent_changes import ROOT,BASE
from scripts.experiment_two_stage import save_rows,control_predictions
from scripts.high_score_weighting import COMBINED,high_score_weights,point_group_scores,alert_metrics,paired_comparison
from scripts.prior_season_profile import PROFILE_FEATURES,build_profiles,augment,profile_matrix
from scripts.time_validation_tuning import build_folds,FitBudget,dump,FOLDS
from nba_dashboard.data import expected_hash

VARIANTS={'basic_prior14':dict(columns=BASE+PROFILE_FEATURES,multiplier=1),
          'combined_prior25_w2':dict(columns=COMBINED+PROFILE_FEATURES,multiplier=2)}
CONTROLS={'basic9':'pred_basic_unweighted','combined20_w2':'pred_combined_w2',
          'baseline':'pred_baseline','frozen9':'pred_frozen9'}
ALL=['baseline','basic9','basic_prior14','combined20_w2','combined_prior25_w2','frozen9']
PAIRS=[('basic9','basic_prior14'),('combined20_w2','combined_prior25_w2')]
NAMES=dict(baseline='五場均值',basic9='原九項（共同訓練）',basic_prior14='原九項＋上季表現',
           combined20_w2='1＋3＋同區＋2倍',combined_prior25_w2='1＋3＋同區＋上季表現＋2倍',frozen9='原九項凍結模型（背景參考）')


def csv_rows(path,rows):
    save_rows(path,[{k:None if isinstance(v,float) and np.isnan(v) else v for k,v in r.items()} for r in rows])


def summarize(period,rows):
    scores=[]; alerts=[]
    for variant in ALL:
        def add(selected,group=None):
            if not selected: return
            stats=point_group_scores([float(r['target_pts']) for r in selected],[float(r['pred_'+variant]) for r in selected])
            scores.extend(dict(period=period,variant=variant,**dict(s,group=group) if group else s)
                          for s in (stats[:1] if group else stats))
        add(rows)
        for flag,group in [(1,'has_prior'),(0,'no_prior')]: add([r for r in rows if int(r['has_prev_season'])==flag],group)
        alerts.append(dict(period=period,variant=variant,**alert_metrics([float(r['target_pts']) for r in rows],[float(r['pred_'+variant]) for r in rows])))
    return scores,alerts


def main():
    out=ROOT/'outputs/week2/prior_season_profile_v9'
    if out.exists(): raise ValueError('Output exists: refuse overwrite and extra fits')
    prior=ROOT/'outputs/week2/same_conference_h2h_v6'; controls=ROOT/'outputs/week2/high_score_weighting_v7'
    source=ROOT/'outputs/nba_2024_25/nba_2024_25_regular_season_player_game_logs.csv'
    source_audit=json.loads(subprocess.check_output([sys.executable,'-X','utf8',str(source.parent/'verify_csv.py')],text=True,encoding='utf-8'))
    inputs=[source,source.parent/'source_rows.json',source.parent/'verify_csv.py',source.parent/'validation.json',
        prior/'features/train.csv',prior/'features/validation.csv',controls/'predictions_CV_pooled.csv',controls/'predictions_February.csv',
        controls/'validation.json',ROOT/'outputs/week2/time_validation_tuning/common_parameters.json']
    protected=[p for p in (ROOT/'outputs/week2').rglob('*') if p.is_file()]
    protected += [ROOT/'streamlit_app.py',ROOT/'nba_dashboard/data.py',ROOT/'nba_dashboard/views.py']
    before={str(p.relative_to(ROOT)):sha(p) for p in set(inputs+protected)}
    meta=json.loads((controls/'validation.json').read_text(encoding='utf-8'))
    for p in inputs[4:6]:
        if sha(p)!=expected_hash(meta['source_sha256'],p.relative_to(ROOT)): raise ValueError('Changed feature source')
    for p in inputs[6:8]:
        if sha(p)!=expected_hash(meta['output_sha256'],p.relative_to(controls)): raise ValueError('Changed control')
    raw=read_csv(source); profiles=build_profiles(raw)
    train,val=[read_csv(p) for p in inputs[4:6]]; validate_development_rows(train,val)
    train,val=[augment(rows,profiles) for rows in [train,val]]
    for rows in [train,val]: rows.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    if (len(train),len(val))!=(13221,3462): raise ValueError('Wrong cohort')
    folds=build_folds(train)
    if [(len(tr),len(va)) for _,tr,va in folds]!=[(4198,4104),(8302,2422),(10724,2497)]: raise ValueError('Wrong folds')
    params=json.loads(inputs[-1].read_text(encoding='utf-8'))['selected']['xgboost']['params']
    out.mkdir(parents=True); (out/'models').mkdir(); (out/'features').mkdir()
    dump(out/'experiment_config.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),variants=VARIANTS,params=params,
        fit_cap=8,folds=FOLDS,common_train_n=13221,cv_n=9023,february_n=3462,profile_season='2024-25',
        profile_cutoff='2025-04-13',target_season='2025-26',target='target_pts',pairs=PAIRS,paired_groups=['all','30+'],
        iterations=2000,seed=42,multiplicity_adjusted=False,test_scoring_allowed=False,auto_promotion=False,
        development_reused=True,exploratory_after_test_observed=True,player_id_encoding=False,
        no_prior='first three NaN, games=0, has=0; not necessarily rookie',rate='sum PTS / sum MIN; positive MIN only'))
    dump(out/'prior_source_audit.json',dict(**source_audit,zero_minutes_excluded=sum(float(r['MIN'])==0 for r in raw),
        positive_minutes_n=sum(float(r['MIN'])>0 for r in raw),profile_players=len(profiles)))
    csv_rows(out/'player_profiles.csv',[dict(player_id=name,**p) for name,p in profiles.items()])
    joins=[dict(player_id=name,match_method='exact' if name in profiles else 'no_prior_record',
        **profiles.get(name,dict(zip(PROFILE_FEATURES,[np.nan,np.nan,np.nan,0,0])))) for name in sorted({r['player_id'] for r in train+val})]
    csv_rows(out/'player_join_audit.csv',joins)
    for split,rows in [('train',train),('validation',val)]: csv_rows(out/f'features/{split}.csv',rows)
    oldcv=read_csv(inputs[6]); oldfeb=read_csv(inputs[7]); budget=FitBudget(8); fit_log=[]; predictions={}
    with threadpool_limits(limits=2):
        for period,tr,va in folds+[('February',train,val)]:
            old=[r for r in oldcv if r['period']==period] if period!='February' else oldfeb
            aligned={name:control_predictions(va,old,column) for name,column in CONTROLS.items()}
            rows=[{k:r[k] for k in ['player_id','game_id','GAMEDATE','target_pts','has_prev_season']}|
                  dict(period=period,**{'pred_'+name:float(p[i]) for name,p in aligned.items()}) for i,r in enumerate(va)]
            y=np.asarray([float(r['target_pts']) for r in tr])
            for variant,spec in VARIANTS.items():
                weights,normalizer=high_score_weights(y,spec['multiplier'])
                model=XGBRegressor(**params); budget.consume()
                model.fit(profile_matrix(tr,spec['columns']),y,sample_weight=weights)
                pred=model.predict(profile_matrix(va,spec['columns'])).astype(float)
                no_labels=[{k:v for k,v in r.items() if not k.startswith('target_')} for r in va]
                fake=[dict(r,target_pts=99999,PTS=99999,MIN=99999) for r in no_labels]
                if not np.isfinite(pred).all() or any(not np.array_equal(pred,model.predict(profile_matrix(x,spec['columns'])).astype(float)) for x in [no_labels,fake]):
                    raise ValueError('Outcome-dependent or nonfinite inference')
                bundle=dict(model=model,columns=spec['columns'],params=params,variant=variant,multiplier=spec['multiplier'],
                    raw_weight_normalizer=normalizer,n_train=len(tr),trained_through=max(r['GAMEDATE'] for r in tr),research_only=True,profile_season='2024-25')
                path=out/'models'/f'{period}_{variant}.joblib'; joblib.dump(bundle,path)
                loaded=joblib.load(path)
                if not np.array_equal(pred,loaded['model'].predict(profile_matrix(no_labels,loaded['columns'])).astype(float)): raise ValueError('Reload differs')
                for r,p in zip(rows,pred): r['pred_'+variant]=float(p)
                fit_log.append(dict(period=period,variant=variant,fit_number=budget.count,n_features=len(spec['columns']),n_train=len(tr),n_validation=len(va),
                    train_high_n=int((y>=30).sum()),multiplier=spec['multiplier'],raw_weight_normalizer=normalizer,effective_weight_mean=float(weights.mean()),
                    train_end=bundle['trained_through'],validation_start=min(r['GAMEDATE'] for r in va)))
            predictions[period]=rows
    predictions['CV_pooled']=sum([predictions[name] for name,_,_ in folds],[])
    scores=[]; alerts=[]; intervals=[]
    for period,rows in predictions.items():
        s,a=summarize(period,rows); scores+=s; alerts+=a
        if period in ['CV_pooled','February']:
            save_rows(out/f'predictions_{period}.csv',rows)
            for first,second in PAIRS:
                for group in ['all','30+']:
                    intervals.append(dict(period=period,first=first,second=second,group=group,**paired_comparison(rows,first,second,group)))
    for name,rows in [('scores',scores),('alert_metrics',alerts),('fit_log',fit_log)]: save_rows(out/f'{name}.csv',rows)
    dump(out/'paired_intervals.json',intervals); make_report(out,scores,intervals,joins)
    if budget.count!=8 or any(sha(ROOT/p)!=h for p,h in before.items()): raise ValueError('Fit budget or protected files changed')
    code=[Path(__file__),ROOT/'scripts/prior_season_profile.py',ROOT/'scripts/high_score_weighting.py',ROOT/'scripts/time_validation_tuning.py']
    dump(out/'validation.json',dict(fits=8,common_train_n=13221,cv_n=9023,february_n=3462,
        profile_join_n=len(joins),matched_players=sum(r['has_prev_season'] for r in joins),missing_players=sum(not r['has_prev_season'] for r in joins),
        model_reload_equal=True,inference_without_or_fake_labels_equal=True,evaluation_unweighted=True,
        production_promoted=False,test_metrics_computed=False,source_sha256=before,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},python=platform.python_version(),
        output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(dict(fits=8,output=str(out),scores=[r for r in scores if r['period'] in ['CV_pooled','February'] and r['group'] in ['all','30+']]),ensure_ascii=False))


def make_report(out,scores,intervals,joins):
    lines=['# 上季球員表現研究','',
        '用2024–25正分鐘比賽建立上季得分、分鐘、總分／總分鐘、出賽場數與有資料指標。跨隊合併，只按完整姓名接合；缺資料三個表現欄留空，場數與指標0，不排除樣本，也不一律稱新秀。',
        f"本季共同樣本球員{len(joins)}位，其中{sum(r['has_prev_season'] for r in joins)}位有上季紀錄，其他保留缺值。",'',
        '原參數固定，兩候選×四期共8fit。30+2倍按本折原實際得分，除以本折平均權重，評估不加權。原九項凍結背景模型訓練13281筆，其餘共同訓練13221筆。','',
        '|期間|模型|整體MAE|RMSE|30+ MAE|<30 MAE|有上季MAE|無上季MAE|','|---|---|---:|---:|---:|---:|---:|---:|']
    for period in ['CV_pooled','February']:
        for variant in ALL:
            get=lambda group:next(r for r in scores if r['period']==period and r['variant']==variant and r['group']==group)
            a=get('all'); vals=[a['mae'],a['rmse']]+[get(g)['mae'] for g in ['30+','under30','has_prior','no_prior']]
            lines.append('|'+period+'|'+NAMES[variant]+'|'+'|'.join(f'{v:.4f}' for v in vals)+'|')
    lines+=['','相同資料、特徵及參數下，各新增方案只與自己的舊模型比較，才能區分上季特徵的貢獻。兩個新方案間另有特徵與加權差異。','',
        '|期間|比較|組別|MAE變化（新−舊）|95%日期區間|','|---|---|---|---:|---|']
    for r in intervals:
        lines.append(f"|{r['period']}|{NAMES[r['first']]} → {NAMES[r['second']]}|{r['group']}|{r['delta_mae']:.4f}|[{r['lower_95']:.4f}, {r['upper_95']:.4f}]|")
    lines+=['','負變化代表改善。區間為日期群組2000次重抽，未做多重比較校正。完整分箱／警示與AP在CSV，AP為得分排序指標，不是機率。',
        '本次結果：上季特徵在兩個方案的CV與二月整體／30+誤差均略降，高分改善較明顯；但二月整體改善區間仍跨過0，尚不足以宣稱穩定升級。',
        '二月無上季紀錄的475筆樣本，原九項MAE 4.1144 → 加上季4.1451；組合2倍4.1018 → 加上季4.1089，均稍退步。沒有上季資料時仍能預測，但不能保證重訓後預測與原模型相同。',
        '組合2倍二月30分警示召回率由13.13%降至11.11%，精確率40.63%降至36.67%；雖然高分場次的得分誤差降低，辨識是否達30分沒有同步改善。CV的警示則略改善，詳細見alert_metrics.csv。',
        '本輪只使用上季固定背景與本季訓練／開發期，不把上季比賽加入本季訓練列；本季五場資格及跨季移動窗重置不變。',
        'CV與二月已多次用於研究，屬日期隔離的假設性回測，季末快照不能證明當時發布版本。沒有重新評分三月／Luka，不依舊測試選模，也沒有替換線上模型。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
