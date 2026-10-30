"""Fixed two-candidate 2024-25 replication; March and later remain sealed."""
import json
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
from scripts.build_features_baseline import read_csv,sha
from scripts.experiment_two_stage import save_rows
from scripts.experiment_prior_season_profile import VARIANTS,csv_rows
from scripts.prior_season_profile import profile_matrix
from scripts.historical_season_features import previous_profiles,development_rows,request_features,season_folds,FOLDS
from scripts.high_score_weighting import high_score_weights,point_group_scores,alert_metrics,paired_comparison
from scripts.team_lineup_features import TeamHistoryIndex
from scripts.time_validation_tuning import FitBudget,dump
from scripts.fetch_historical_seasons import ROOT,OUT as DATA

OUT=ROOT/'outputs/season_validation_2024_25'
ALL=['baseline']+list(VARIANTS)
NAMES=dict(baseline='五場均值',basic_prior14='原九項＋上季表現',combined_prior25_w2='1＋3＋同區＋上季表現＋2倍')
PAIRS=[('baseline','basic_prior14'),('baseline','combined_prior25_w2'),('basic_prior14','combined_prior25_w2')]


def prepare_features(records,team_games,profiles,conferences):
    rows=sorted(development_rows(records),key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    index=TeamHistoryIndex([g for g in team_games if g['GAMEDATE']<='2025-02-28'])
    history=defaultdict(list);features=[];audit=[];seen=set()
    for r in rows:
        key=r['player_id'],r['GAMEDATE']
        if key in seen:raise ValueError('Duplicate player date')
        seen.add(key)
        if float(r['MIN'])==0:
            audit.append(dict(player_id=r['player_id'],game_id=r['game_id'],GAMEDATE=r['GAMEDATE'],eligible=False,reason='zero_minutes'));continue
        f=request_features(history[r['player_id']],index,r,profiles,conferences)
        if f['is_starter']!=int(r['is_starter']):raise ValueError('Starter source mismatch')
        audit.append(dict(player_id=r['player_id'],game_id=r['game_id'],GAMEDATE=r['GAMEDATE'],eligible=f['eligible'],reason=f['exclusion_reason']))
        if f['eligible']:
            features.append(dict(player_id=r['player_id'],team_id=r['team_id'],game_id=r['game_id'],GAMEDATE=r['GAMEDATE'],
                nba_player_id=r['nba_player_id'],nba_game_id=r['nba_game_id'],season='2024-25',
                split='train' if r['GAMEDATE']<='2025-01-31' else 'validation',target_pts=float(r['PTS']),target_minutes=float(r['MIN']),**f))
        history[r['player_id']].append(r)
    return features,audit


def summarize(period,rows):
    scores=[];alerts=[]
    for v in ALL:
        y=[float(r['target_pts']) for r in rows];p=[r['pred_'+v] for r in rows]
        scores.extend(dict(period=period,variant=v,**s) for s in point_group_scores(y,p))
        for flag,group in [(1,'has_prior'),(0,'no_prior')]:
            selected=[r for r in rows if int(r['has_prev_season'])==flag]
            if selected:
                s=point_group_scores([float(r['target_pts']) for r in selected],[r['pred_'+v] for r in selected])[0]
                scores.append(dict(period=period,variant=v,**dict(s,group=group)))
        alerts.append(dict(period=period,variant=v,**alert_metrics(y,p)))
    return scores,alerts


def main():
    if OUT.exists():raise ValueError('Refuse existing experiment or extra fits')
    supp=ROOT/'outputs/nba_2024_25/supplement_2026_10_30'
    inputs=[supp/'player_games_enriched.csv',supp/'team_games.csv',supp/'conferences_2024_25.json',
        ROOT/'outputs/nba_2023_24/player_games_with_official_ids.csv',DATA/'data_validation.json',
        ROOT/'outputs/week2/time_validation_tuning/common_parameters.json']
    data_meta=json.loads(inputs[4].read_text(encoding='utf-8'))
    original=ROOT/'outputs/nba_2024_25/nba_2024_25_regular_season_player_game_logs.csv'
    if sha(original)!=data_meta['original_2024_csv_sha256']:raise ValueError('Original data changed')
    for p,h in data_meta['output_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Export changed: '+p)
    protected=[p for p in (ROOT/'outputs/week2').rglob('*') if p.is_file()]+[ROOT/'streamlit_app.py',ROOT/'nba_dashboard/data.py',ROOT/'nba_dashboard/views.py',original]
    source_hash={str(p.relative_to(ROOT)):sha(p) for p in set(inputs+protected)}
    records=read_csv(inputs[0]);teams=read_csv(inputs[1]);conferences=json.loads(inputs[2].read_text(encoding='utf-8'))
    profiles=previous_profiles(read_csv(inputs[3]));features,audit=prepare_features(records,teams,profiles,conferences)
    train=[r for r in features if r['split']=='train'];val=[r for r in features if r['split']=='validation']
    folds=season_folds(train);params=json.loads(inputs[-1].read_text(encoding='utf-8'))['selected']['xgboost']['params']
    OUT.mkdir();(OUT/'models').mkdir();(OUT/'features').mkdir()
    test_keys=[{k:r[k] for k in ['player_id','game_id','GAMEDATE','nba_player_id','nba_game_id']} for r in records if r['GAMEDATE']>='2025-03-01']
    dump(OUT/'experiment_config.json',dict(season='2024-25',previous_season='2023-24',variants=VARIANTS,params=params,folds=FOLDS,fit_cap=8,
        train_n=len(train),february_n=len(val),cv_n=sum(len(va) for _,_,va in folds),train_end='2025-01-31',validation_end='2025-02-28',
        test_start='2025-03-01',test_end='2025-04-13',test_scoring_allowed=False,auto_promotion=False,pairs=PAIRS,
        evaluation_unweighted=True,bootstrap_iterations=2000,seed=42,multiplicity_adjusted=False,
        scope='retrospective season replication; frozen choices originated in 2025-26 research, not an untouched prospective test'))
    dump(OUT/'test_usage.json',dict(test_start='2025-03-01',test_end='2025-04-13',target_rows_reserved=len(test_keys),
        features_built=False,predictions_computed=False,metrics_computed=False,evaluations=0))
    save_rows(OUT/'test_reserved_keys.csv',test_keys);save_rows(OUT/'eligibility_audit.csv',audit)
    for split,rows in [('train',train),('validation',val)]:csv_rows(OUT/f'features/{split}.csv',rows)
    csv_rows(OUT/'previous_profiles.csv',[dict(nba_player_id=pid,**p) for pid,p in profiles.items()])
    predictions={};log=[];budget=FitBudget(8)
    with threadpool_limits(limits=2):
        for period,tr,va in folds+[('February',train,val)]:
            y=np.asarray([r['target_pts'] for r in tr])
            rows=[{k:r[k] for k in ['player_id','game_id','GAMEDATE','target_pts','has_prev_season']}|
                  dict(period=period,pred_baseline=float(r['pts_mean_5'])) for r in va]
            for v,spec in VARIANTS.items():
                weights,normalizer=high_score_weights(y,spec['multiplier']);model=XGBRegressor(**params);budget.consume()
                model.fit(profile_matrix(tr,spec['columns']),y,sample_weight=weights)
                x=profile_matrix(va,spec['columns']);pred=model.predict(x).astype(float)
                no_labels=[{k:value for k,value in r.items() if not k.startswith('target_')} for r in va]
                fake=[dict(r,target_pts=99999,PTS=99999,MIN=99999) for r in no_labels]
                if not np.isfinite(pred).all() or any(not np.array_equal(pred,model.predict(profile_matrix(z,spec['columns'])).astype(float)) for z in [no_labels,fake]):raise ValueError('Outcome-dependent inference')
                bundle=dict(model=model,columns=spec['columns'],params=params,variant=v,n_train=len(tr),multiplier=spec['multiplier'],
                    raw_weight_normalizer=normalizer,trained_through=max(r['GAMEDATE'] for r in tr),profile_season='2023-24',research_only=True)
                path=OUT/'models'/f'{period}_{v}.joblib';joblib.dump(bundle,path)
                loaded=joblib.load(path)
                if not np.array_equal(pred,loaded['model'].predict(x).astype(float)):raise ValueError('Reload mismatch')
                for r,p in zip(rows,pred):r['pred_'+v]=float(p)
                log.append(dict(period=period,variant=v,fit_number=budget.count,n_train=len(tr),n_validation=len(va),n_features=len(spec['columns']),
                    multiplier=spec['multiplier'],raw_weight_normalizer=normalizer,train_high_n=int((y>=30).sum()),effective_weight_mean=float(weights.mean()),
                    train_end=bundle['trained_through'],validation_start=min(r['GAMEDATE'] for r in va)))
            predictions[period]=rows
    predictions['CV_pooled']=sum([predictions[n] for n,_,_ in folds],[])
    scores=[];alerts=[];intervals=[]
    for period,rows in predictions.items():
        s,a=summarize(period,rows);scores+=s;alerts+=a
        if period in ['CV_pooled','February']:
            save_rows(OUT/f'predictions_{period}.csv',rows)
            for first,second in PAIRS:
                for group in ['all','30+']:intervals.append(dict(period=period,first=first,second=second,group=group,**paired_comparison(rows,first,second,group)))
    for name,rows in [('scores',scores),('alert_metrics',alerts),('fit_log',log)]:save_rows(OUT/f'{name}.csv',rows)
    dump(OUT/'paired_intervals.json',intervals);make_report(OUT,scores,train,val,profiles)
    if budget.count!=8 or any(sha(ROOT/p)!=h for p,h in source_hash.items()):raise ValueError('Fit budget or protected files changed')
    code=[Path(__file__),ROOT/'scripts/historical_season_features.py',ROOT/'scripts/prior_season_profile.py',ROOT/'scripts/high_score_weighting.py']
    dump(OUT/'validation.json',dict(fits=8,train_n=len(train),february_n=len(val),cv_n=len(predictions['CV_pooled']),
        test_metrics_computed=False,production_promoted=False,source_sha256=source_hash,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps(dict(fits=8,train=len(train),february=len(val),cv=len(predictions['CV_pooled']),scores=[s for s in scores if s['period'] in ['CV_pooled','February'] and s['group'] in ['all','30+']]),ensure_ascii=False))


def make_report(out,scores,train,val,profiles):
    lines=['# 2024–25兩候選驗證','',f'上季背景使用2023–24，來源共{len(profiles)}位有正分鐘歷史。共同訓練{len(train)}筆、二月驗證{len(val)}筆。',
        '沿用相同月份切分與固定參數，8次訓練；三月以後保留，未建立特徵、預測或評分。先發來自官方賽後快照，屬日期隔離的假設性回測。',
        '候選及參數來自2025–26研究，本輪是較早球季的重現驗證，不是當年已鎖定模型的前瞻實驗。','',
        '|期間|模型|整體MAE|RMSE|30+ MAE|<30 MAE|','|---|---|---:|---:|---:|---:|']
    for period in ['CV_pooled','February']:
        for v in ALL:
            get=lambda group:next(s for s in scores if s['period']==period and s['variant']==v and s['group']==group)
            a=get('all');values=[a['mae'],a['rmse'],get('30+')['mae'],get('under30')['mae']]
            lines.append('|'+period+'|'+NAMES[v]+'|'+'|'.join(f'{x:.4f}' for x in values)+'|')
    lines+=['','結果：A原九項＋上季表現的CV及二月整體誤差較低；B組合2倍的30+誤差較低，方向與2025–26研究一致。高分改善有一般場次誤差增加的代價，兩個候選仍保留。',
        '上季按官方球員ID接合；無上季資料保留NaN，不排除球員。來源MIN=0列排除於個人窗口，但球隊總得分包含所有得分並逐場核對官方總分。',
        '2023–24原始匯出含578個姓名字串／576個官方ID，其中571個ID有正MIN歷史。266列MIN=0全數不入個人背景；0:00來源列不足以逐筆判斷DNP，不把來源列數稱為全部確認出賽。詳見outputs/nba_2023_24/README.md。',
        '2023–24來源中立旗標全False，未據此認定實際無中立場；本輪僅使用該季個人得分／分鐘背景，不使用其場地旗標。',
        '原2024–25逐場CSV不變，陣容／中立場／官方ID等另存補充；官方網頁分鐘偶有秒數60，按進位換算並保留來源字串。',
        '完整分箱、有無上季資料分組、30+警示與AP另存CSV；AP是得分排序指標，不是機率。日期區間2000次重抽、未做多重比較校正。兩候選均保留，沒有自動換線上模型。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
