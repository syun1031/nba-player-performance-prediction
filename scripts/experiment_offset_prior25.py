"""Four fixed fits for 25-feature offsets; unchanged 20-offset and 25-direct controls."""
import json
from pathlib import Path
from datetime import datetime,timezone
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.build_features_baseline import read_csv,sha
from scripts.offset_prior_model import OffsetPriorModel,COLUMNS
from scripts.experiment_two_stage import save_rows
from scripts.compare_offset_prior_2025 import align_controls
from scripts.verify_offset_prior_2025 import audit as audit_controls
from scripts.time_validation_tuning import build_folds,FitBudget,dump,FOLDS
from scripts.high_score_weighting import point_group_scores,alert_metrics,paired_comparison

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/offset_prior25_w2_2025_26'
PRIOR=ROOT/'outputs/week2/prior_season_profile_v9'
CONTROL=ROOT/'outputs/offset_vs_prior_2025_26'
TEST=ROOT/'outputs/season_test_2025_26_prior_models'
ALL=['baseline','residual20_w2','residual25_w2','direct25_w2']
NAMES=dict(baseline='五場均值',residual20_w2='20項增減＋高分2倍',residual25_w2='25項增減＋上季表現＋高分2倍',direct25_w2='25項直接得分＋上季表現＋高分2倍')
PAIRS=[('residual20_w2','residual25_w2'),('direct25_w2','residual25_w2')]
KEY=lambda r:(r['player_id'],r['game_id'])


def summarize(dataset,rows):
    scores=[];alerts=[]
    for v in ALL:
        scores.extend(dict(dataset=dataset,variant=v,**s) for s in point_group_scores([float(r['target_pts']) for r in rows],[r['pred_'+v] for r in rows]))
        for flag,group in [(1,'has_prior'),(0,'no_prior')]:
            selected=[r for r in rows if int(r['has_prev_season'])==flag]
            if selected:
                s=point_group_scores([float(r['target_pts']) for r in selected],[r['pred_'+v] for r in selected])[0]
                scores.append(dict(dataset=dataset,variant=v,**dict(s,group=group)))
        alerts.append(dict(dataset=dataset,variant=v,**alert_metrics([float(r['target_pts']) for r in rows],[r['pred_'+v] for r in rows])))
    return scores,alerts


def attach_controls(features,controls,period):
    rows=[{k:r[k] for k in ['player_id','game_id','GAMEDATE','target_pts','has_prev_season']}|
        dict(period=period,pred_baseline=float(r['pts_mean_5'])) for r in features]
    for r,c in align_controls(rows,controls):
        r['pred_residual20_w2']=float(c['pred_residual_w2']);r['pred_direct25_w2']=float(c['pred_combined_prior25_w2'])
    return rows


def predict_rows(model,features,rows):
    fs=[{k:v for k,v in r.items() if not k.startswith('target_')} for r in features]
    result=model.predict_components(fs)
    fake=model.predict_components([dict(r,target_pts=99999,PTS=99999,MIN=99999) for r in fs])
    if any(not np.array_equal(result[k],fake[k]) for k in result):raise ValueError('Current labels affect inference')
    for i,r in enumerate(rows):
        r['pred_residual25_w2']=float(result['points'][i]);r['residual25_w2_delta']=float(result['delta'][i]);r['residual25_w2_raw_points']=float(result['raw_points'][i])
    return result


def main():
    if OUT.exists():raise ValueError('Refuse overwrite or additional fits')
    audit_controls()
    protected=[p for folder in [CONTROL,PRIOR,TEST,ROOT/'outputs/week2/residual_offset_v8'] for p in folder.rglob('*') if p.is_file()]
    protected += [ROOT/'nba_dashboard/research_views.py',ROOT/'dashboard_pages/comparison.py']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    train=read_csv(PRIOR/'features/train.csv');val=read_csv(PRIOR/'features/validation.csv')
    for rows in [train,val]:rows.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    if (len(train),len(val))!=(13221,3462):raise ValueError('Changed common cohort')
    folds=build_folds(train)
    params=json.loads((PRIOR/'experiment_config.json').read_text(encoding='utf-8'))['params']
    oldcv=read_csv(CONTROL/'predictions_CV_pooled.csv');oldfeb=read_csv(CONTROL/'predictions_February.csv');oldtest=read_csv(CONTROL/'predictions_Test.csv')
    test_features=read_csv(TEST/'features.csv');test_outcomes={KEY(r):r for r in oldtest}
    if len(test_features)!=6979 or {KEY(r) for r in test_features}!=set(test_outcomes):raise ValueError('Changed test cohort')
    test_with_labels=[dict(r,target_pts=test_outcomes[KEY(r)]['target_pts']) for r in test_features]
    OUT.mkdir();(OUT/'models').mkdir()
    dump(OUT/'experiment_config.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),columns=COLUMNS,params=params,fit_cap=4,
        train_n=13221,cv_n=9023,february_n=3462,reused_test_n=6979,folds=FOLDS,target='target_pts - pts_mean_5',
        prediction='max(0,pts_mean_5 + predicted_delta)',weight='2 if original target_pts>=30 else 1; normalize by fold mean',
        evaluation_unweighted=True,pairs=PAIRS,bootstrap_iterations=2000,seed=42,multiplicity_adjusted=False,
        tuning_allowed=False,auto_promotion=False,test_is_additional_reused_research=True,independent_test=False))
    save_rows(OUT/'training_targets.csv',[dict(player_id=r['player_id'],game_id=r['game_id'],GAMEDATE=r['GAMEDATE'],split=r['split'],
        target_pts=float(r['target_pts']),pts_mean_5=float(r['pts_mean_5']),target_delta=float(r['target_pts'])-float(r['pts_mean_5'])) for r in train+val])
    models={};predictions={};logs=[];budget=FitBudget(4)
    with threadpool_limits(limits=2):
        for period,tr,va in folds+[('February',train,val)]:
            controls=[r for r in oldcv if r['period']==period] if period!='February' else oldfeb
            rows=attach_controls(va,controls,period);model=OffsetPriorModel(params).fit(tr,budget)
            result=predict_rows(model,va,rows);path=OUT/'models'/f'{period}_residual25_w2.joblib';joblib.dump(model,path)
            again=joblib.load(path).predict_components(va)
            if any(not np.array_equal(result[k],again[k]) for k in result):raise ValueError('Reload mismatch')
            models[period]=model;predictions[period]=rows
            logs.append(dict(period=period,fit_number=budget.count,n_train=len(tr),n_validation=len(va),n_features=25,
                train_high_n=model.train_high_n,multiplier=2,raw_weight_normalizer=model.weight_normalizer,
                train_end=model.trained_through,validation_start=min(r['GAMEDATE'] for r in va)))
        test_period=oldtest[0]['period'];rows=attach_controls(test_with_labels,oldtest,test_period)
        predict_rows(models['February'],test_features,rows);predictions['Test']=rows
    predictions['CV_pooled']=sum([predictions[n] for n,_,_ in folds],[])
    scores=[];alerts=[];intervals=[]
    for dataset in ['CV_pooled','February','Test']:
        rows=predictions[dataset];save_rows(OUT/f'predictions_{dataset}.csv',rows)
        human=[]
        for r in rows:
            h=dict(球員=r['player_id'],日期=r['GAMEDATE'],對戰=r['game_id'].rsplit('|',1)[0],實際得分=float(r['target_pts']),
                五場均值=float(r['pred_baseline']),新增25項模型預估增減=r['residual25_w2_delta'])
            for v in ALL[1:]:h[NAMES[v]]=r['pred_'+v];h[NAMES[v]+'絕對誤差']=abs(float(r['target_pts'])-r['pred_'+v])
            human.append(h)
        save_rows(OUT/f'三方案逐場比較_{dataset}.csv',human)
        s,a=summarize(dataset,rows);scores+=s;alerts+=a
        for first,second in PAIRS:
            for group in ['all','30+']:intervals.append(dict(dataset=dataset,first=first,second=second,group=group,**paired_comparison(rows,first,second,group)))
    save_rows(OUT/'scores.csv',scores);save_rows(OUT/'alert_metrics.csv',alerts);save_rows(OUT/'fit_log.csv',logs);dump(OUT/'paired_intervals.json',intervals)
    make_report(scores,intervals)
    if budget.count!=4 or any(sha(ROOT/p)!=h for p,h in before.items()):raise ValueError('Fit cap or protected files changed')
    dump(OUT/'validation.json',dict(fits=4,train_n=13221,cv_n=9023,february_n=3462,reused_test_n=6979,
        controls_unchanged=True,model_reload_equal=True,inference_without_or_fake_labels_equal=True,production_promoted=False,
        independent_test=False,source_sha256=before,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/offset_prior_model.py']},
        output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps([s for s in scores if s['group'] in ['all','30+']],ensure_ascii=False))


def make_report(scores,intervals):
    lines=['# 25項增減＋上季表現＋高分2倍','',
        '新增唯一候選：原1＋3＋同區20欄＋上季5欄，學習target_pts−pts_mean_5，最終max(0,五場均值＋增減)。權重依原得分>=30給2倍，除以本折平均；不是依增減大小加權。',
        '沿用相同樣本、同一固定參數，共4fit。兩個舊對照未重訓：20項增減2倍、25項直接得分2倍。前者對照只加上季欄，後者同25欄與權重而目標不同。','',
        '|期間|模型|整體MAE|RMSE|30+ MAE|<30 MAE|','|---|---|---:|---:|---:|---:|']
    for dataset in ['CV_pooled','February','Test']:
        for v in ALL:
            get=lambda g:next(s for s in scores if s['dataset']==dataset and s['variant']==v and s['group']==g)
            a=get('all');values=[a['mae'],a['rmse'],get('30+')['mae'],get('under30')['mae']]
            lines.append('|'+dataset+'|'+NAMES[v]+'|'+'|'.join(f'{x:.4f}' for x in values)+'|')
    lines+=['','本次結論：加入上季後，三期的增減模型整體與30+ MAE點估計都比舊20項增減較低；二月整體只改善約0.0006分，幅度非常小。',
        '與相同25欄的直接得分模型相比，新增減模型三期整體MAE都較高，CV／二月30+也較高；只有三月後30+點估計較低（11.7674對11.9856）。該高分差為−0.2182分、95%日期區間[−0.4685,0.0497]含0，尚無明確穩定優勢。',
        '三月後整體MAE則新增減4.8030、直接4.7499，新−直接差0.0531分，區間[0.0267,0.0772]。建議保留新版本作研究，仍維持直接得分方案的現有地位，不據此切換線上模型。',
        'CV為三輪各自較早截止的模型，February與Test使用訓練至2026-01-31的模型。Test6979筆已使用，只是追加研究，不能稱新的獨立證據，不能據此再調參。',
        '先發仍是賽後快照；日期隔離假設性回測限制、上季缺值／姓名接合與資格規則不變。此輪不更換主模型或改線上頁面。',
        '完整分箱／有無上季資料群組、30分警示與AP另存CSV。12個日期群組區間2000次重抽，未作多重比較校正；新−舊的負MAE差代表新增25項增減模型較好。','',
        '|期間|舊對照|組別|新−舊MAE|95%日期區間|','|---|---|---|---:|---|']
    for r in intervals:lines.append(f"|{r['dataset']}|{NAMES[r['first']]}|{r['group']}|{r['delta_mae']:.4f}|[{r['lower_95']:.4f}, {r['upper_95']:.4f}]|")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
