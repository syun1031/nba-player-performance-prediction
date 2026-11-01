"""Build two-season fixed25 routing display from immutable saved predictions."""
from pathlib import Path
from datetime import datetime,timezone
import json
import pandas as pd
from nba_dashboard.research_data import load_research
from nba_dashboard.routing_data import apply_route,MODELS,comparison_metrics,load_routing
from scripts.build_features_baseline import sha

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'outputs/dashboard_routing'
    if out.exists():raise ValueError('Refuse overwrite existing display release')
    old=load_research(ROOT);out.mkdir()
    manifest=dict(schema=1,threshold=25,primary='route25',models=MODELS,built_at_utc=datetime.now(timezone.utc).isoformat(),datasets={},histories={},
        new_fits=0,new_model_inferences=0,fixed_user_policy=True,history_display_strategy_changed=True,frozen_model_files_changed=False,
        note='Retrospective date-isolated backtest with postgame lineup snapshots; both test periods already used; no new independent test')
    history_paths={'2025-26':'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv','2024-25':'outputs/nba_2024_25/supplement_2026_10_30/player_games_enriched.csv'}
    for season,periods in old.predictions.items():
        manifest['datasets'][season]={}
        source=ROOT/history_paths[season];h=pd.read_csv(source)
        h['actual_minutes']=h['MIN'] if season=='2025-26' else h['actual_minutes']
        history=h[['PLAYER','TEAM','MATCHUP','GAMEDATE','MIN','PTS','actual_minutes']].copy()
        history_path=out/('history_'+season.replace('-','_')+'.csv');history.to_csv(history_path,index=False)
        manifest['histories'][season]=dict(file=history_path.name,sha256=sha(history_path),source=history_paths[season],source_sha256=sha(source),n=len(history))
        for period,frame in periods.items():
            f=apply_route(frame);path=out/(season.replace('-','_')+'_'+period+'.csv');f.to_csv(path,index=False)
            info=dict(file=path.name,sha256=sha(path),source='outputs/dashboard_comparison/'+old.manifest['datasets'][season][period]['file'],source_sha256=old.manifest['datasets'][season][period]['sha256'],
                n=len(f),route_counts=f.route25_model.value_counts().to_dict(),metrics={})
            for group,threshold in [('all',None),('20+',20),('30+',30)]:info['metrics'][group]={r['model_variant']:{k:r[k] for k in ['n','mae','rmse','mean_residual']} for r in comparison_metrics(f,threshold=threshold).to_dict('records')}
            manifest['datasets'][season][period]=info
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    verified=load_routing(ROOT)
    print(json.dumps({s:{p:dict(n=len(f),switched=int(f.pred_baseline.ge(25).sum()),mae=comparison_metrics(f,['route25']).iloc[0].mae) for p,f in periods.items()} for s,periods in verified.predictions.items()},ensure_ascii=False))

if __name__=='__main__':main()
