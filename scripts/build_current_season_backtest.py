"""Build a separately labelled 2026-27 hypothetical backtest; never fit models."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
from nba_dashboard.current_season import backtest_rows,write_csv,BASE_COLUMNS,FOLDER,load_current
from nba_dashboard.future_data import load_future,sha
from nba_dashboard.future_inference import load_models


def build(root):
    root=Path(root);bundle=load_future(root);models=load_models(root,bundle)
    predictions,excluded=backtest_rows(bundle,models)
    out=root/FOLDER;out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'predictions.csv',predictions,BASE_COLUMNS)
    write_csv(out/'excluded.csv',excluded,['nba_player_id','player_id','team_id','nba_game_id','game_id','GAMEDATE','reason'])
    m=bundle['manifest']
    manifest=dict(schema=1,season='2026-27',kind='date_isolated_hypothetical_backtest',
        history_version=m['version'],history_manifest_sha256=sha(root/'outputs/future_prediction/manifest.json'),
        through=m['through'],last_completed_date=m.get('last_completed_date'),collected_at_utc=m['collected_at_utc'],
        generated_at_utc=datetime.now(timezone.utc).isoformat(),models=m['models'],
        eligible_rows=len(predictions),excluded_rows=len(excluded),source_rows=len(bundle['raw_history']),
        lineup_source_kind='historical_postgame',pregame_verified=False,
        files={name:sha(out/name) for name in ['predictions.csv','excluded.csv']})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    load_current(root)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    build(p.parse_args().root)
