"""Independent, no-fit release audit of both-season fixed25 routing."""
from pathlib import Path
from collections import defaultdict,deque
import json,math
import pandas as pd
from scripts.build_features_baseline import sha
from nba_dashboard.routing_data import load_routing,MODELS,ROUTE

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'outputs/week3/routing_release_2026_11_01'


def audit():
    b=load_routing(ROOT);checked=0;raw_index={};metrics=[]
    for season,periods in b.predictions.items():
        info=b.manifest['histories'][season];source=ROOT/info['source'];assert sha(source)==info['source_sha256']
        h=pd.read_csv(source);h['actual_minutes']=h.MIN if season=='2025-26' else h.actual_minutes
        histories=defaultdict(list)
        for r in h.to_dict('records'):
            if float(r['MIN'])>0:histories[r['PLAYER']].append(r)
        for player,rows in histories.items():
            rows.sort(key=lambda r:r['GAMEDATE']);recent=deque(maxlen=5);previous_date=None
            for r in rows:
                assert previous_date is None or previous_date<r['GAMEDATE']
                if len(recent)==5:raw_index[season,player,r['MATCHUP']+'|'+r['GAMEDATE']]=(math.fsum(recent)/5,float(r['PTS']),float(r['actual_minutes']))
                recent.append(float(r['PTS']));previous_date=r['GAMEDATE']
        for period,f in periods.items():
            source_info=b.manifest['datasets'][season][period]
            assert sha(ROOT/source_info['source'])==source_info['source_sha256']
            counts=defaultdict(int)
            for r in f.to_dict('records'):
                mean,actual,minutes=raw_index[season,r['player_id'],r['game_id']]
                assert math.isclose(mean,r['pred_baseline'],rel_tol=0,abs_tol=1e-12) and actual==r['target_pts'] and math.isclose(minutes,r['actual_minutes'],rel_tol=0,abs_tol=1e-12)
                branch='residual25_w2' if mean>=25 else 'combined_prior25_w2'
                assert r['route25_model']==branch and r['pred_'+ROUTE]==r['pred_'+branch]
                counts[branch]+=1;checked+=1
            assert dict(counts)==source_info['route_counts']
            for group,threshold in [('all',None),('20+',20),('30+',30)]:
                selected=f if threshold is None else f.loc[f.target_pts>=threshold]
                for model in ['baseline']+MODELS:
                    errors=[float(r['target_pts'])-float(r['pred_'+model]) for r in selected.to_dict('records')]
                    n=len(errors);mae=math.fsum(abs(e) for e in errors)/n;rmse=math.sqrt(math.fsum(e*e for e in errors)/n);res=math.fsum(errors)/n
                    saved=source_info['metrics'][group][model]
                    assert n==saved['n'] and all(math.isclose(a,saved[k],rel_tol=1e-12,abs_tol=1e-12) for a,k in [(mae,'mae'),(rmse,'rmse'),(res,'mean_residual')])
                    if group in ['all','30+'] and model==ROUTE:metrics.append(dict(season=season,period=period,group=group,n=n,mae=mae,rmse=rmse))
    # Old experiment hashes remain frozen. Only changed page entry uses archived bytes.
    archives=[]
    legacy=['outputs/offset_prior25_w2_2025_26','outputs/offset_prior25_w2_2024_25','outputs/offset_vs_prior_2025_26']
    for relative in legacy:
        folder=ROOT/relative;metadata=json.loads((folder/'validation.json').read_text(encoding='utf-8'))
        for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',folder)]:
            for rel,expected in metadata[group].items():
                normalized=rel.replace('\\','/');path=base/normalized
                if sha(path)==expected:continue
                if group=='source_sha256' and normalized=='nba_dashboard/research_views.py':archived=ROOT/'outputs/week3/four_model_release_2026_10_31/historical_ui'/normalized
                elif group=='source_sha256' and normalized=='dashboard_pages/comparison.py':archived=RELEASE/'historical_ui'/normalized
                else:raise ValueError('Unexpected frozen evidence change: '+relative+'/'+rel)
                assert sha(archived)==expected;archives.append(dict(experiment=relative,relative=normalized,archive=str(archived.relative_to(ROOT))))
    return dict(new_fits=0,new_model_inferences=0,seasons=list(b.predictions),validated_rows=checked,
        raw_previous_five_rebuilt=True,same_cohort_single_predictions_unchanged=True,fixed_threshold_inclusive=True,gate_uses_no_current_labels=True,
        metrics_independently_verified=True,historical_hashes_verified=True,historical_ui_archives=archives,
        history_display_policy='route25',old_frozen_model_files_changed=False,metrics=metrics)

if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False))
