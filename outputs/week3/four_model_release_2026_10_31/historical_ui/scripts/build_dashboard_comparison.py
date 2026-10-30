"""Export only existing predictions into a small independently checked display bundle."""
from datetime import datetime,timezone
import json
from pathlib import Path
import pandas as pd
from scripts.build_features_baseline import sha
from scripts.time_validation_tuning import dump
from nba_dashboard.research_data import MODELS,EXPECTED,research_metrics,load_research

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'outputs/dashboard_comparison';out.mkdir(exist_ok=True)
    specs={
        '2025-26':{'CV':'outputs/week2/prior_season_profile_v9/predictions_CV_pooled.csv','validation':'outputs/week2/prior_season_profile_v9/predictions_February.csv','test':'outputs/season_test_2025_26_prior_models/predictions_test.csv'},
        '2024-25':{'CV':'outputs/season_validation_2024_25_three_models/predictions_CV_pooled.csv','validation':'outputs/season_validation_2024_25_three_models/predictions_February.csv','test':'outputs/season_test_2024_25_three_models/predictions_test.csv'}}
    manifest=dict(schema=1,models=MODELS,built_at_utc=datetime.now(timezone.utc).isoformat(),datasets={},
        primary_model_changed=False,new_fits=0,new_predictions=0,
        note='Existing immutable research predictions only; original nine means the same-cohort retrained reference, not a new deployed primary')
    for season,periods in specs.items():
        manifest['datasets'][season]={}
        if season=='2025-26':
            history_path=ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv'
            h=pd.read_csv(history_path);h['game_id']=h['MATCHUP']+'|'+h['GAMEDATE']
            minutes=h.rename(columns={'PLAYER':'player_id','MIN':'actual_minutes'})[['player_id','game_id','actual_minutes']]
        for period,relative in periods.items():
            source=ROOT/relative;frame=pd.read_csv(source)
            if 'actual_minutes' not in frame:
                if season=='2024-25':
                    history_path=ROOT/'outputs/nba_2024_25/supplement_2026_10_30/player_games_enriched.csv'
                    h=pd.read_csv(history_path);minutes=h[['player_id','game_id','actual_minutes']]
                frame=frame.merge(minutes,on=['player_id','game_id'],how='left',validate='one_to_one')
            if 'period' not in frame:frame['period']=period
            columns=['player_id','game_id','GAMEDATE','target_pts','actual_minutes','period','pred_baseline']+['pred_'+v for v in MODELS]
            frame=frame[columns];filename=season.replace('-','_')+'_'+period+'.csv';path=out/filename
            frame.to_csv(path,index=False)
            info=dict(file=filename,sha256=sha(path),source=relative,source_sha256=sha(source),
                minutes_source=str(history_path.relative_to(ROOT)) if 'actual_minutes' not in pd.read_csv(source,nrows=0).columns else relative,
                metrics={})
            for group,threshold in [('all',None),('20+',20),('30+',30)]:
                info['metrics'][group]={r['model_variant']:{k:r[k] for k in ['n','mae','rmse','mean_residual']} for r in research_metrics(frame,['baseline']+MODELS,threshold).to_dict('records')}
            manifest['datasets'][season][period]=info
    dump(out/'manifest.json',manifest)
    load_research(ROOT)
    print(json.dumps({s:{p:EXPECTED[s][p][2] for p in periods} for s,periods in specs.items()},ensure_ascii=False))


if __name__=='__main__':main()
