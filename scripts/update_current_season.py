"""One daily update; caller publishes only after this entire process succeeds."""
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import csv
import subprocess
import sys
from scripts.actions_report import report_error
from scripts.build_future_prediction_bundle import build as build_future
from scripts.build_current_season_backtest import build as build_backtest
from nba_dashboard.future_data import load_future
from nba_dashboard.current_season import load_current


def collection_day(now):return now.astimezone(ZoneInfo('America/New_York')).date().isoformat()


def update(root,*,now=None):
    root=Path(root).resolve();now=now or datetime.now(timezone.utc);day=collection_day(now)
    if not '2026-10-01'<=day<='2027-06-30':raise ValueError('Outside 2026-27 collection window')
    pointer=root/'outputs/nba_2026_27/latest_success.json'
    before=pointer.read_bytes() if pointer.exists() else None
    old_manifest=root/'outputs/future_prediction/manifest.json'
    old=json.loads(old_manifest.read_text(encoding='utf-8')) if old_manifest.exists() else {}
    subprocess.run([sys.executable,'-X','utf8','-m','scripts.fetch_current_season','--through',day],cwd=root,check=True)
    if not pointer.exists() or pointer.read_bytes()==before:raise ValueError('No fresh successful snapshot')
    latest=json.loads(pointer.read_text(encoding='utf-8'));name=latest['snapshot']
    snapshot=(pointer.parent/name).resolve()
    if Path(name).name!=name or not snapshot.is_relative_to(pointer.parent.resolve()):raise ValueError('Invalid snapshot path')
    m=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    if not m['complete'] or m['errors'] or m['through']!=day:raise ValueError('Invalid new collection')
    if m['completed_games_collected']<old.get('completed_games',0):raise ValueError('Completed game count regressed')
    if old.get('completed_games',0):
        def game_ids(path):
            with path.open(encoding='utf-8-sig',newline='') as file:return {r['nba_game_id'] for r in csv.DictReader(file)}
        if game_ids(old_manifest.parent/'team_games.csv')-game_ids(snapshot/'team_games.csv'):
            raise ValueError('Previously completed games disappeared')
    # The disposable Actions checkout publishes neither package until both pass.
    build_future(snapshot,replace=True)
    build_backtest(root)
    history=load_future(root);backtest=load_current(root)
    if history['manifest']['version']!=name or backtest['manifest']['history_version']!=name:raise ValueError('Mixed package versions')
    summary=dict(through=day,history_version=name,completed_games=history['manifest']['completed_games'],
        eligible_rows=backtest['manifest']['eligible_rows'],excluded_rows=backtest['manifest']['excluded_rows'])
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':
    try:update(Path(__file__).resolve().parents[1])
    except subprocess.CalledProcessError:raise  # the collector already reported its own errors
    except Exception as exc:
        report_error('Daily update stopped',f'{type(exc).__name__}: {exc}');raise
