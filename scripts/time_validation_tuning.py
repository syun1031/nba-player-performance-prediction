"""Three expanding date folds, a fixed search budget, and February-only selection."""
import argparse
import csv
import json
import itertools
import time
from pathlib import Path
from datetime import date
import importlib.metadata

import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

if __package__:
    from .compare_models import BASIC, SETTINGS, metrics, validate_development_rows
    from .build_features_baseline import read_csv, save_csv, sha
    from .merge_game_rosters import roster_index, matchup_info
    from .teammate_encoder import TeammateEncoder, validate_names
else:
    from compare_models import BASIC, SETTINGS, metrics, validate_development_rows
    from build_features_baseline import read_csv, save_csv, sha
    from merge_game_rosters import roster_index, matchup_info
    from teammate_encoder import TeammateEncoder, validate_names

ROOT=Path(__file__).resolve().parents[1]
GROUPS=['basic','starter','teammates']
FOLDS=[('F1','2025-11-30','2025-12-01','2025-12-31'),
       ('F2','2025-12-31','2026-01-01','2026-01-15'),
       ('F3','2026-01-15','2026-01-16','2026-01-31')]


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def build_folds(rows):
    seen=set()
    for r in rows:
        day=r['GAMEDATE']
        key=r['player_id'],r['game_id']
        if date.fromisoformat(day).isoformat()!=day or not '2025-10-21'<=day<='2026-01-31' or key in seen:
            raise ValueError('Invalid or duplicated training date/sample')
        seen.add(key)
    folds=[]
    for name,end,start,last in FOLDS:
        train=[r for r in rows if r['GAMEDATE']<=end]
        val=[r for r in rows if start<=r['GAMEDATE']<=last]
        if not train or not val or max(r['GAMEDATE'] for r in train)>=min(r['GAMEDATE'] for r in val):
            raise ValueError('Empty or overlapping fold')
        folds.append((name,train,val))
    return folds


def rebuild_teammates(rows,index):
    result=[]
    for row in rows:
        matchup,day=row['game_id'].rsplit('|',1)
        team=matchup.split()[0]
        roster=index.get((day,team))
        if day!=row['GAMEDATE'] or roster is None or roster['pair']!=matchup_info(matchup,team)[0]:
            raise ValueError('Roster key or opponent mismatch')
        own=row['player_id']
        if own not in roster['players'] or float(row['is_starter']) not in (0.,1.):
            raise ValueError('Player missing from roster or invalid starter flag')
        is_starter=int(float(row['is_starter']))
        if int(own in roster['starters'])!=is_starter:
            raise ValueError('Starter flag disagrees with source roster')
        names=sorted(n for n in roster['starters'] if n!=own)
        validate_names(names)
        if len(names)!=5-is_starter:
            raise ValueError('Wrong number of teammates')
        result.append(dict(row,_teammates=names))
    return result


def prepare_features(train,val):
    encoder=TeammateEncoder().fit([r['_teammates'] for r in train])
    columns=dict(basic=list(BASIC),starter=BASIC+['is_starter'],
                 teammates=BASIC+['is_starter']+encoder.get_feature_names_out())
    X={}; y={}; unknown=None
    for split,rows in [('train',train),('validation',val)]:
        base=np.asarray([[float(r[c]) for c in columns['starter']] for r in rows])
        encoded=np.asarray(encoder.transform([r['_teammates'] for r in rows]))
        target=np.asarray([float(r['target_pts']) for r in rows])
        if not len(rows) or not np.isfinite(base).all() or not np.isfinite(target).all():
            raise ValueError('Empty or nonfinite model data; no silent row removal')
        if any(sum(v)!=5-int(float(r['is_starter'])) or r['player_id'] in r['_teammates']
               for r,v in zip(rows,encoded)):
            raise ValueError('Teammate identity/count contract violated')
        X[split]=dict(basic=base[:,:8],starter=base,teammates=np.column_stack([base,encoded]))
        y[split]=target
        if split=='train' and encoded[:,-1].any():
            raise ValueError('Training unknown count is not zero')
        if split=='validation':
            unknown=encoded[:,-1]
    return dict(encoder=encoder,columns=columns,X=X,y=y,unknown_validation=unknown)


def parameter_grid():
    grids={'linear':[dict(parameter_id='linear_01',params=dict(SETTINGS['linear']))]}
    grids['tree']=[dict(parameter_id=f'tree_{i:02d}',params=dict(SETTINGS['tree'],max_depth=d,min_samples_leaf=l))
                   for i,(d,l) in enumerate(itertools.product([4,6,8],[30,60]),1)]
    grids['xgboost']=[dict(parameter_id=f'xgboost_{i:02d}',params=dict(SETTINGS['xgboost'],max_depth=d,n_estimators=n,min_child_weight=w))
                      for i,(d,n,w) in enumerate(itertools.product([2,3],[150,300],[1,10]),1)]
    return grids


class FitBudget:
    def __init__(self,limit=144):
        self.limit=limit
        self.count=0

    def consume(self):
        if self.count>=self.limit:
            raise RuntimeError('Fixed fit budget exhausted; no extra search allowed')
        self.count+=1


def fit_candidate(name,params,group,prepared,budget):
    model=(make_pipeline(StandardScaler(),LinearRegression(**params)) if name=='linear'
           else DecisionTreeRegressor(**params) if name=='tree' else XGBRegressor(**params))
    budget.consume()
    model.fit(prepared['X']['train'][group],prepared['y']['train'])
    if name=='linear' and int(model.steps[0][1].n_samples_seen_)!=len(prepared['y']['train']):
        raise ValueError('Scaler was fitted outside training sample')
    prediction=model.predict(prepared['X']['validation'][group])
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions')
    return model,prediction


def pooled_metrics(parts):
    return metrics(np.concatenate([np.asarray(y) for y,_ in parts]),
                   np.concatenate([np.asarray(p) for _,p in parts]))


def choose_common_parameters(results,grids):
    selected={}
    for name,grid in grids.items():
        candidates=[]
        for setting in grid:
            score=next(r for r in results if r['model']==name and r['features']=='starter'
                       and r['parameter_id']==setting['parameter_id'])
            p=setting['params']
            simpler=((p['max_depth'],-p['min_samples_leaf']) if name=='tree'
                     else (p['max_depth'],p['n_estimators'],-p['min_child_weight']) if name=='xgboost' else ())
            candidates.append(((score['mae'],score['rmse'],*simpler),setting))
        selected[name]=dict(min(candidates,key=lambda item:item[0])[1])
    return selected


def stratified_interval(actual,old,new,dates,strata,iterations=2000,seed=42):
    actual,old,new=[np.asarray(a,dtype=float) for a in (actual,old,new)]
    dates=np.asarray(dates); strata=np.asarray(strata)
    if not (actual.ndim==old.ndim==new.ndim==1 and len(actual)==len(old)==len(new)==len(dates)==len(strata)):
        raise ValueError('Unaligned paired interval data')
    if not len(actual) or not np.isfinite(np.concatenate([actual,old,new])).all() or iterations<1:
        raise ValueError('Invalid paired interval inputs')
    delta=np.abs(new-actual)-np.abs(old-actual)
    rng=np.random.default_rng(seed)
    numerator=np.zeros(iterations); denominator=np.zeros(iterations)
    for stratum in np.unique(strata):
        mask=strata==stratum
        groups,index=np.unique(dates[mask],return_inverse=True)
        if len(groups)<2:
            raise ValueError('Need at least two dates per stratum')
        sums=np.bincount(index,weights=delta[mask]); counts=np.bincount(index)
        draws=rng.multinomial(len(groups),[1/len(groups)]*len(groups),size=iterations)
        numerator+=draws@sums; denominator+=draws@counts
    lo,hi=np.quantile(numerator/denominator,[.025,.975])
    return dict(delta_mae=float(delta.mean()),lower_95=float(lo),upper_95=float(hi),
                delta_rmse=metrics(actual,new)['rmse']-metrics(actual,old)['rmse'],iterations=iterations,seed=seed)


def effect_diagnostics(name,paired,subgroups):
    effects=[r for r in paired if r['model']==name and r['old_features']=='starter']
    improvements=sum(r['delta_mae']<0 for r in effects if r['period'] in ['F1','F2','F3'])
    feb=next(r for r in effects if r['period']=='February')
    unknown={r['model_variant']:r['mae'] for r in subgroups
             if r['period']=='February' and r['group']=='has_unknown'}
    delta=unknown[name+'_teammates']-unknown[name+'_starter']
    return dict(fold_improvements=improvements,only_one_or_no_improving_fold=improvements<=1,
                february_direction='worse' if feb['lower_95']>0 else 'better' if feb['upper_95']<0 else 'uncertain',
                unknown_february_delta_mae=float(delta),unknown_february_worse=delta>0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/week2/time_validation_tuning')
    args=parser.parse_args()
    out=args.output_dir
    if out.exists():
        raise FileExistsError('Keep previous runs intact; choose a new --output-dir to rerun')
    source=ROOT/'outputs/week1/team_lineup_features'
    roster_path=ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_game_rosters.csv'
    reference_path=ROOT/'outputs/week1/teammate_encoding/teammate_encoder.json'
    input_paths=[source/'train.csv',source/'validation.csv',roster_path,reference_path]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in input_paths}
    train=read_csv(source/'train.csv'); feb=read_csv(source/'validation.csv')
    validate_development_rows(train,feb)
    assert (len(train),len(feb))==(13281,3462)
    roster_rows=[r for r in read_csv(roster_path) if r['GAMEDATE']<='2026-02-28']
    index=roster_index(roster_rows)
    train=rebuild_teammates(train,index); feb=rebuild_teammates(feb,index)
    for rows in [train,feb]:
        rows.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    folds=build_folds(train)
    assert [(len(a),len(b)) for _,a,b in folds]==[(4258,4104),(8362,2422),(10784,2497)]
    grids=parameter_grid()
    manifest=[dict(fold=f,model=n,parameter_id=p['parameter_id'],features=g,params=p['params'])
              for f,_,_ in folds for n,settings in grids.items() for p in settings for g in GROUPS]
    assert len(manifest)==135
    out.mkdir(parents=True)
    (out/'encoders').mkdir(); (out/'models').mkdir()
    save_csv(out/'training_sample_manifest.csv',
             [{k:r[k] for k in ['player_id','game_id','GAMEDATE']} for r in train],
             ['player_id','game_id','GAMEDATE'])
    started=time.perf_counter()
    dump(out/'experiment_config.json',dict(spec_version='time-tuning-v1',seed=42,folds=FOLDS,parameter_grid=grids,
         feature_groups=GROUPS,cv_run_manifest=manifest,cv_fit_count=135,february_fit_count=9,fit_limit=144,
         parameter_selection_features='starter',selection_metric_order=['pooled_mae','pooled_rmse','simpler_parameters'],
         early_stopping=False,extra_search=False,refit_through='2026-01-31',test_evaluation=False,
         versions={n:importlib.metadata.version(n) for n in ['numpy','scikit-learn','xgboost','matplotlib','joblib']},
         source_sha256=hashes))
    budget=FitBudget(); search_results=[]; cv_data={}; cohort=[]; coverage=[]

    def execute(name,setting,group,prepared,scope):
        t=time.perf_counter()
        try:
            model,pred=fit_candidate(name,setting['params'],group,prepared,budget)
            score=metrics(prepared['y']['validation'],pred)
            record=dict(fit_number=budget.count,scope=scope,model=name,parameter_id=setting['parameter_id'],
                        features=group,train_n=len(prepared['y']['train']),**score,seconds=time.perf_counter()-t)
            with (out/'fit_log.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
            return model,pred,record
        except Exception as exc:
            dump(out/'failure.json',dict(scope=scope,model=name,setting=setting,features=group,
                                        fit_count=budget.count,error=repr(exc),action='stop_no_extra_search'))
            raise

    with threadpool_limits(limits=2):
        for i,(fold,tr,va) in enumerate(folds):
            prepared=prepare_features(tr,va)
            assert len(prepared['encoder'].classes_)==[284,325,345][i]
            assert int((prepared['unknown_validation']>0).sum())==[748,328,133][i]
            dump(out/'encoders'/(fold+'.json'),dict(encoder=prepared['encoder'].to_dict(),feature_columns=prepared['columns']))
            for role,data in [('train',tr),('validation',va)]:
                cohort.extend(dict(fold=fold,role=role,player_id=r['player_id'],game_id=r['game_id'],GAMEDATE=r['GAMEDATE']) for r in data)
            coverage.append(dict(period=fold,train_n=len(tr),validation_n=len(va),
                                 vocabulary_size=len(prepared['encoder'].classes_),
                                 unknown_samples=int((prepared['unknown_validation']>0).sum())))
            cv_data[fold]=dict(rows=va,y=prepared['y']['validation'],unknown=prepared['unknown_validation'],
                               pred={'baseline':np.asarray([float(r['pts_mean_5']) for r in va])})
            for name,settings in grids.items():
                for setting in settings:
                    for group in GROUPS:
                        model,pred,record=execute(name,setting,group,prepared,fold)
                        cv_data[fold]['pred'][setting['parameter_id']+'_'+group]=pred
                        search_results.append(record)
            print(f'{fold} complete: {budget.count}/144 fits; vocabulary={len(prepared["encoder"].classes_)}',flush=True)
        assert budget.count==135
        save_csv(out/'fold_manifest.csv',cohort,['fold','role','player_id','game_id','GAMEDATE'])
        save_csv(out/'search_results.csv',search_results,list(search_results[0]))
        pooled=[]
        for name,settings in grids.items():
            for setting in settings:
                for group in GROUPS:
                    key=setting['parameter_id']+'_'+group
                    pooled.append(dict(model=name,parameter_id=setting['parameter_id'],features=group,
                                       **pooled_metrics([(v['y'],v['pred'][key]) for v in cv_data.values()])))
        save_csv(out/'pooled_search_results.csv',pooled,list(pooled[0]))
        common=choose_common_parameters(pooled,grids)
        dump(out/'common_parameters.json',dict(selected=common,basis='starter pooled F1-F3 MAE, then RMSE, then simplicity',
                                              february_used_for_parameter_selection=False))
        cv_header=['fold','player_id','game_id','GAMEDATE','target_pts','teammate_unknown_count']
        cv_keys=list(next(iter(cv_data.values()))['pred'])
        cv_export=[]
        for fold,data in cv_data.items():
            for j,r in enumerate(data['rows']):
                row={c:r[c] for c in ['player_id','game_id','GAMEDATE','target_pts']}
                row.update(fold=fold,teammate_unknown_count=int(data['unknown'][j]))
                row.update({'pred_'+k:float(data['pred'][k][j]) for k in cv_keys})
                cv_export.append(row)
        save_csv(out/'predictions_time_validation.csv',cv_export,cv_header+['pred_'+k for k in cv_keys])
        assert len(cv_export)==9023
        prepared=prepare_features(train,feb)
        reference=json.loads(reference_path.read_text(encoding='utf-8'))
        assert prepared['encoder'].to_dict()==reference
        dump(out/'encoders/February.json',dict(encoder=prepared['encoder'].to_dict(),feature_columns=prepared['columns']))
        coverage.append(dict(period='February',train_n=len(train),validation_n=len(feb),vocabulary_size=356,
                             unknown_samples=int((prepared['unknown_validation']>0).sum())))
        assert coverage[-1]['unknown_samples']==505
        feb_pred={'baseline':np.asarray([float(r['pts_mean_5']) for r in feb])}
        final_scores=[dict(model='baseline',features='basic',parameter_id='none',split='validation',
                           **metrics(prepared['y']['validation'],feb_pred['baseline']))]
        train_scores=[]
        reload_checks=[]
        for name,setting in common.items():
            for group in GROUPS:
                model,pred,record=execute(name,setting,group,prepared,'February')
                key=name+'_'+group; feb_pred[key]=pred
                final_scores.append(dict(model=name,features=group,parameter_id=setting['parameter_id'],
                                         split='validation',**metrics(prepared['y']['validation'],pred)))
                train_scores.append(dict(model=name,features=group,parameter_id=setting['parameter_id'],split='train',
                     **metrics(prepared['y']['train'],model.predict(prepared['X']['train'][group]))))
                bundle=dict(model=model,feature_columns=prepared['columns'][group],
                            encoder=prepared['encoder'].to_dict() if group=='teammates' else None,
                            params=setting['params'],fit_end_date='2026-01-31')
                model_path=out/'models'/(key+'.joblib')
                joblib.dump(bundle,model_path,compress=3)
                saved=joblib.load(model_path)
                np.testing.assert_allclose(saved['model'].predict(prepared['X']['validation'][group]),pred,rtol=1e-12,atol=1e-12)
                assert saved['feature_columns']==prepared['columns'][group]
                if saved['encoder'] is not None:
                    restored=TeammateEncoder.from_dict(saved['encoder'])
                    assert restored.transform([feb[0]['_teammates']])==prepared['encoder'].transform([feb[0]['_teammates']])
                reload_checks.append(key)
        assert budget.count==144
    print('February complete: 144/144 fits; test data unused',flush=True)
    save_csv(out/'coverage.csv',coverage,list(coverage[0]))
    save_csv(out/'february_metrics.csv',final_scores+train_scores,list(final_scores[0]))
    feb_export=[]
    for j,r in enumerate(feb):
        row={c:r[c] for c in ['player_id','game_id','GAMEDATE','target_pts']}
        row['teammate_unknown_count']=int(prepared['unknown_validation'][j])
        row.update({'pred_'+k:float(v[j]) for k,v in feb_pred.items()})
        feb_export.append(row)
    save_csv(out/'predictions_february.csv',feb_export,list(feb_export[0]))
    periods={}
    for fold,data in cv_data.items():
        periods[fold]=dict(rows=data['rows'],y=data['y'],unknown=data['unknown'],
                          pred={'baseline':data['pred']['baseline']},strata=[fold]*len(data['rows']))
        for name,setting in common.items():
            for group in GROUPS:
                periods[fold]['pred'][name+'_'+group]=data['pred'][setting['parameter_id']+'_'+group]
    periods['pooled']=dict(rows=[r for data in cv_data.values() for r in data['rows']],
                          y=np.concatenate([d['y'] for d in cv_data.values()]),
                          unknown=np.concatenate([d['unknown'] for d in cv_data.values()]),
                          pred={key:np.concatenate([periods[f]['pred'][key] for f in cv_data]) for key in feb_pred},
                          strata=[f for f,d in cv_data.items() for _ in d['rows']])
    periods['February']=dict(rows=feb,y=prepared['y']['validation'],unknown=prepared['unknown_validation'],
                            pred=feb_pred,strata=['February']*len(feb))
    paired=[]; subgroup=[]; selected_period_scores=[]
    for period,data in periods.items():
        for key,pred in data['pred'].items():
            selected_period_scores.append(dict(period=period,model_variant=key,**metrics(data['y'],pred)))
            for label,mask in [('known_only',data['unknown']==0),('has_unknown',data['unknown']>0)]:
                if mask.any():
                    subgroup.append(dict(period=period,group=label,model_variant=key,**metrics(data['y'][mask],pred[mask])))
        for name in common:
            for old,new in [('basic','starter'),('starter','teammates')]:
                paired.append(dict(period=period,model=name,old_features=old,new_features=new,
                    **stratified_interval(data['y'],data['pred'][name+'_'+old],data['pred'][name+'_'+new],
                                          [r['GAMEDATE'] for r in data['rows']],data['strata'])))
    save_csv(out/'paired_differences.csv',paired,list(paired[0]))
    save_csv(out/'unknown_coverage_metrics.csv',subgroup,list(subgroup[0]))
    save_csv(out/'selected_parameters_period_metrics.csv',selected_period_scores,list(selected_period_scores[0]))
    winner=min(final_scores,key=lambda r:(r['mae'],r['rmse'],['baseline','linear','tree','xgboost'].index(r['model']),GROUPS.index(r['features'])))
    winner_key=winner['model']+'_'+winner['features']
    model_path=out/'models'/(winner_key+'.joblib') if winner['model']!='baseline' else None
    dump(out/'selected_candidate.json',dict(status='locked_for_future_test_not_tested',selection_basis='February MAE then RMSE; original tie rules',
        result=winner,params=common[winner['model']]['params'] if winner['model']!='baseline' else {},
        feature_columns=prepared['columns'][winner['features']] if winner['model']!='baseline' else ['pts_mean_5'],
        model_file=str(model_path.relative_to(out)) if model_path else None,model_sha256=sha(model_path) if model_path else None,
        encoder_file='encoders/February.json' if winner['features']=='teammates' else None,
        encoder_sha256=sha(out/'encoders/February.json') if winner['features']=='teammates' else None,
        training_samples=13281,fit_through='2026-01-31',validation_samples=3462,
        versions=json.loads((out/'experiment_config.json').read_text(encoding='utf-8'))['versions'],
        experiment_config='experiment_config.json',experiment_config_sha256=sha(out/'experiment_config.json'),
        training_sample_manifest='training_sample_manifest.csv',training_manifest_sha256=sha(out/'training_sample_manifest.csv'),
        validation_sample_manifest='predictions_february.csv',validation_manifest_sha256=sha(out/'predictions_february.csv'),
        source_sha256=hashes,test_evaluated=False,information_policy='Historical postgame lineup; hypothetical date-isolated backtest'))
    # Independent file readback: all 135 search scores and all February predictions.
    cv_disk=read_csv(out/'predictions_time_validation.csv')
    for r in search_results:
        subset=[s for s in cv_disk if s['fold']==r['scope']]
        key='pred_'+r['parameter_id']+'_'+r['features']
        score=metrics([float(s['target_pts']) for s in subset],np.asarray([float(s[key]) for s in subset]))
        assert all(score[k]==r[k] for k in ['n','mae','rmse'])
    feb_disk=read_csv(out/'predictions_february.csv')
    for r in final_scores:
        key='pred_'+('baseline' if r['model']=='baseline' else r['model']+'_'+r['features'])
        score=metrics([float(s['target_pts']) for s in feb_disk],np.asarray([float(s[key]) for s in feb_disk]))
        assert all(score[k]==r[k] for k in ['n','mae','rmse'])
    keys=[(r['player_id'],r['game_id']) for r in cv_disk]
    assert len(keys)==len(set(keys))==9023
    assert not set(keys).intersection((r['player_id'],r['game_id']) for r in feb_disk)
    fit_log=[json.loads(line) for line in (out/'fit_log.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(fit_log)==144 and [r['fit_number'] for r in fit_log]==list(range(1,145))
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for ax,metric in zip(axes,['mae','rmse']):
        for j,group in enumerate(GROUPS):
            values=[next(r[metric] for r in final_scores if r['model']==n and r['features']==group) for n in common]
            bars=ax.bar(np.arange(3)+(j-1)*.25,values,.25,label=group)
            ax.bar_label(bars,fmt='%.3f',fontsize=8)
        ax.axhline(final_scores[0][metric],ls='--',color='black',label='Last-five mean')
        ax.set_xticks(np.arange(3),list(common)); ax.set_ylabel(metric.upper()+' (points; lower is better)')
        ax.set_title('February: common parameters per model'); ax.legend(fontsize=8)
    fig.savefig(out/'february_comparison.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    for name in common:
        data=[next(r for r in paired if r['model']==name and r['period']==p and r['old_features']=='starter')
              for p in ['F1','F2','F3','February']]
        ax.plot(range(4),[r['delta_mae'] for r in data],marker='o',label=name)
    ax.axhline(0,color='black',ls='--'); ax.set_xticks(range(4),['F1: December','F2: Jan 1-15','F3: Jan 16-31','February'])
    ax.set_ylabel('MAE: teammates minus starter (negative is better)'); ax.legend()
    ax.set_title('Incremental teammate effect with common parameters')
    fig.savefig(out/'teammate_effect_by_period.png',dpi=180); plt.close(fig)
    names={'baseline':'五場平均','linear':'線性迴歸','tree':'決策樹','xgboost':'XGBoost'}
    report=['# 時間驗證與有限調參結果','',
        '2026-10-23。依事先規劃完成 135 次時間驗證訓練與九次二月擬合，共 144 次，沒有追加格點。只使用原訓練／驗證的目標得分，未開啟 test.csv、未評分測試。', '',
        f"依既定二月 MAE／RMSE 規則鎖定候選：{names[winner['model']]}／{winner['features']}，MAE {winner['mae']:.4f}、RMSE {winner['rmse']:.4f} 分。候選與參數見 selected_candidate.json，尚未作最終測試。",'',
        '## 各輪與共用參數','',
        'F1：4258 筆訓練／4104 筆十二月驗證；F2：8362／2422 筆一月上半；F3：10784／2497 筆一月下半。驗證互斥，共 9023 筆。每輪隊友類別 284／325／345，僅由各輪訓練資料建立。', '',
        '共用參數只按 starter 九項特徵的三輪合併 MAE／RMSE 選擇，二月未參與選參；各模型的三組特徵使用同一參數，並非各組特徵各自最佳化。','']
    for name,setting in common.items():
        report.append(f"- {names[name]}：`{json.dumps(setting['params'],ensure_ascii=False,sort_keys=True)}`。")
    report+=['','## 二月驗證（3462 筆相同樣本）','','| 模型 | 特徵 | MAE | RMSE |','| --- | --- | ---: | ---: |']
    for r in final_scores:
        report.append(f"| {names[r['model']]} | {r['features']} | {r['mae']:.4f} | {r['rmse']:.4f} |")
    report+=['','## 隊友編碼的額外效果','','下列均為 teammates 減 starter，負值才是改善；已排除自己是否先發與參數不同的影響。pooled 是三輪逐筆合併，不是平均三個 RMSE。', '',
             '| 模型 | 時段 | MAE 差值 | RMSE 差值 | 探索性 MAE 差值 95% 區間 |','| --- | --- | ---: | ---: | --- |']
    for r in paired:
        if r['old_features']=='starter':
            report.append(f"| {names[r['model']]} | {r['period']} | {r['delta_mae']:+.4f} | {r['delta_rmse']:+.4f} | [{r['lower_95']:+.4f}, {r['upper_95']:+.4f}] |")
    report+=['','## 解讀','']
    for name in common:
        effects=[r for r in paired if r['model']==name and r['old_features']=='starter']
        negative=sum(r['delta_mae']<0 for r in effects if r['period'] in ['F1','F2','F3'])
        feb_effect=next(r for r in effects if r['period']=='February')
        diagnostic=effect_diagnostics(name,paired,subgroup)
        note=dict(uncertain='區間含零，尚無穩定改善證據',worse='區間全部為正，二月顯示退步',
                  better='區間全部為負，仍需配合跨時段結果及最終測試判讀')[diagnostic['february_direction']]
        caveat='；最多只有一輪改善，跨時段方向不一致' if diagnostic['only_one_or_no_improving_fold'] else ''
        unknown_note=f"；含未知隊友的二月 MAE 差異 {diagnostic['unknown_february_delta_mae']:+.4f} 分"
        report.append(f"- {names[name]}：隊友編碼在前三輪 {negative}/3 輪 MAE 下降；二月差異 {feb_effect['delta_mae']:+.4f} 分，{note}{caveat}{unknown_note}。")
    previous_path=ROOT/'outputs/week2/model_comparison/metrics.csv'
    if previous_path.exists():
        previous=[r for r in read_csv(previous_path) if r['split']=='validation']
        old_best=min(previous,key=lambda r:(float(r['mae']),float(r['rmse'])))
        report+=['', '## 與初次比較的差異','',
            f"初次比較的最低二月 MAE 為 {float(old_best['mae']):.4f}（{names[old_best['model']]}／{old_best['features']}）；本次鎖定候選為 {winner['mae']:.4f}，差異 {winner['mae']-float(old_best['mae']):+.4f} 分。調參不保證二月分數更低。",
            '本次先按三輪 starter 成績選共用參數，再只在事先規劃的九個二月候選內排序；不是把歷次全部二月結果放在一起再選最低。初次結果完整保留，不因這個差異追加搜尋。']
    report+=['','## 限制與後續','',
        '- 二月先前已看過，仍是開發／選模資料，不能稱為獨立確認或最終泛化成效。各模型只在一月底以前資料擬合，沒有合併二月重訓。',
        '- 選參後的三輪表現含選擇偏差；日期群組重抽 2000 次只作探索參考，未處理跨日相依或多重比較。合併區間在各輪內分層抽日期。',
        '- 分組誤差見 unknown_coverage_metrics.csv。未知隊友不刪樣本、不擴充類別；未知人數在訓練皆為零，模型沒有學到未知球員的效果。',
        '- 歷史名單來自賽後快照，仍屬日期隔離的假設性研究，不能證明歷史賽前已取得先發，也不提供因果或先發缺席效果。',
        '- linear 的類別與自己是否先發存在線性相依，不將個別隊友係數解讀為因果效果。',
        '- 按預定規則保存最低二月誤差候選；穩定性限制不會事後改寫選模排序。下一步先完成殘差及球員案例，再對鎖定候選與預先列明比較對象執行保留測試，不依測試結果調參。','',
        '## 檔案與重現','',
        '- experiment_config.json：搜尋範圍、執行清單、版本與來源雜湊；在正式訓練前保存。',
        '- fold_manifest.csv、encoders/：各輪樣本角色、類別與特徵順序。',
        '- fit_log.jsonl、search_results.csv、pooled_search_results.csv：完整訓練紀錄與搜尋結果。',
        '- predictions_time_validation.csv：45 個候選與基準的逐筆驗證預測；predictions_february.csv：九個候選與基準的二月預測。',
        '- common_parameters.json：只依 starter 三輪結果選出的共用參數。',
        '- february_metrics.csv、selected_parameters_period_metrics.csv、paired_differences.csv、coverage.csv、unknown_coverage_metrics.csv：評估與覆蓋。',
        '- models/、selected_candidate.json：九個二月候選模型及鎖定候選；只載入信任的 joblib 檔案。',
        '- validation.json：逐筆匯出分數、模型重載、來源／輸出雜湊與訓練預算驗證。','',
        '執行 `.venv/Scripts/python.exe scripts/time_validation_tuning.py`。預設目錄若已存在會拒絕覆寫；另用 --output-dir 指定新目錄才可重新執行，須另外記錄重跑理由。','']
    (out/'README.md').write_text('\n'.join(report),encoding='utf-8')
    dump(out/'validation.json',dict(verified=True,total_fits=budget.count,cv_fits=135,february_fits=9,
        source_files_unchanged=True,source_sha256=hashes,test_split_files_opened=False,test_metrics_computed=False,
        cv_sample_count=9023,february_sample_count=3462,exported_scores_verified=145,
        reloaded_model_predictions_verified=reload_checks,common_parameters_selected_from='starter_F1_F2_F3_only',
        final_candidate=winner,seconds=time.perf_counter()-started,
        output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='validation.json'}))
    print(json.dumps(dict(verified=True,fits=budget.count,common_parameters=common,winner=winner,
                         february_scores=final_scores,paired=paired),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
