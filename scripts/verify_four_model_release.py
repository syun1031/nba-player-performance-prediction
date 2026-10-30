"""Read-only current-release audit, retaining the exact historical UI evidence."""
import json
from pathlib import Path
import joblib
import numpy as np
from scripts.build_features_baseline import read_csv,sha
from scripts.prior_season_profile import profile_matrix
from scripts.offset_prior_model import COLUMNS
from scripts.verify_two_stage import require,score
from scripts.verify_offset_prior_2024 import audit as audit_2024
from nba_dashboard.research_data import load_research,MODELS

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'outputs/week3/four_model_release_2026_10_31'

def audit():
    result24=audit_2024()
    out=ROOT/'outputs/offset_prior25_w2_2025_26'
    metadata=json.loads((out/'validation.json').read_text(encoding='utf-8'))
    archives=[]
    for group,base in [('source_sha256',ROOT),('code_sha256',ROOT),('output_sha256',out)]:
        for relative,h in metadata[group].items():
            p=base/relative
            if sha(p)!=h:
                require(group=='source_sha256' and Path(relative).as_posix()=='nba_dashboard/research_views.py','Unexpected frozen evidence change '+relative)
                archived=RELEASE/'historical_ui'/relative
                require(sha(archived)==h,'Historical UI snapshot mismatch')
                archives.append(relative)
    source=ROOT/'outputs/week2/prior_season_profile_v9'
    training=read_csv(source/'features/train.csv');val=read_csv(source/'features/validation.csv')
    features={ (r['player_id'],r['game_id']):r for r in training+val+read_csv(ROOT/'outputs/season_test_2025_26_prior_models/features.csv')}
    for dataset,n in [('CV_pooled',9023),('February',3462),('Test',6979)]:
        rows=read_csv(out/f'predictions_{dataset}.csv')
        require(len(rows)==len({(r['player_id'],r['game_id']) for r in rows})==n,'25-26 prediction cohort')
        for period in sorted({r['period'] for r in rows}):
            subset=[r for r in rows if r['period']==period];fold='February' if dataset=='Test' else period
            model=joblib.load(out/'models'/f'{fold}_residual25_w2.joblib')
            fs=[features[r['player_id'],r['game_id']] for r in subset]
            require(model.columns==COLUMNS and model.trained_through<min(r['GAMEDATE'] for r in subset),'25-26 model cutoff')
            delta=model.model.predict(profile_matrix(fs,COLUMNS)).astype(float)
            expected=np.maximum(np.array([float(f['pts_mean_5']) for f in fs])+delta,0)
            require(np.array_equal(expected,[float(r['pred_residual25_w2']) for r in subset]),'25-26 independent offset predictions')
    bundle=load_research(ROOT)
    for season,periods in bundle.predictions.items():
        for period,frame in periods.items():
            info=bundle.manifest['datasets'][season][period]
            for name in ['source','offset_source']:require(sha(ROOT/info[name])==info[name+'_sha256'],'Display source hash')
            offset={ (r['player_id'],r['game_id']):r for r in read_csv(ROOT/info['offset_source'])}
            for r in frame.to_dict('records'):
                original=offset[r['player_id'],r['game_id']]
                require(r['GAMEDATE']==original['GAMEDATE'] and r['target_pts']==float(original['target_pts']) and r['pred_residual25_w2']==float(original['pred_residual25_w2']),'Display exact prediction join')
    return dict(models=MODELS,seasons=list(bundle.predictions),new_fits=0,model_2024=result24,
        historical_2025_hashes_verified=True,historical_ui_archives_used=archives,
        model_2025_components_verified=True,display_sources_verified=True,primary_model_changed=False)

if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False))
