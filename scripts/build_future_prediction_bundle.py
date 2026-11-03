"""Package one successful current-season collection; no fitting or inference.

Usage: python -m scripts.build_future_prediction_bundle --snapshot <folder>
Existing packages are archived by hash before explicit --replace.
"""
import argparse,csv,json,shutil,hashlib
from pathlib import Path
from datetime import datetime,timezone
from scripts.merge_game_rosters import ALIASES
from nba_dashboard.future_data import sha,MODEL_PATHS,load_future

ROOT=Path(__file__).resolve().parents[1]


def build(snapshot,*,replace=False):
    src=Path(snapshot).resolve();m=json.loads((src/'manifest.json').read_text(encoding='utf-8'))
    if m['season']!='2026-27' or not m['complete'] or m['errors']:raise ValueError('Not a successful current-season collection')
    for n,digest in m['file_sha256'].items():
        if Path(n).name!=n or sha(src/n)!=digest:raise ValueError('Source CSV mismatch')
    out=ROOT/'outputs/future_prediction'
    if out.exists() and not replace:raise ValueError('Package exists; use --replace to archive and replace')
    staging=ROOT/'outputs'/('future_prediction_staging_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    staging.mkdir(parents=True,exist_ok=False)
    for name in ['player_games.csv','team_games.csv','pending_schedule.csv','conferences_2026_27.json','daily_index.json']:
        shutil.copyfile(src/name,staging/name)
    if (src/'status_resolution.json').exists():shutil.copyfile(src/'status_resolution.json',staging/'status_resolution.json')
    id_by_name={}
    for r in json.loads((ROOT/'outputs/nba_2025_26/source_rows.json').read_text(encoding='utf-8')):
        name=ALIASES.get(r['values'][0],r['values'][0]);pid=r['links'][0].rsplit('/',1)[-1]
        if name in id_by_name and id_by_name[name]!=pid:raise ValueError('Ambiguous old name')
        id_by_name[name]=pid
    with (src/'previous_season_profiles.csv').open(encoding='utf-8-sig',newline='') as f:profiles=list(csv.DictReader(f))
    for r in profiles:r['nba_player_id']=id_by_name[r['player_id']]
    with (staging/'previous_season_profiles.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(profiles[0]));w.writeheader();w.writerows(profiles)
    pending_dates=[]
    with (staging/'pending_schedule.csv').open(encoding='utf-8-sig') as f:
        pending_dates=sorted({r['GAMEDATE'] for r in csv.DictReader(f)})
    manifest=dict(schema=1,season='2026-27',version=src.name,through=m['through'],
        collected_at_utc=m['collected_at_utc'],player_rows=m['player_rows'],team_rows=m['team_rows'],
        completed_games=m['completed_games_collected'],last_completed_date=m['last_game_date'],
        reconciled_game_ids=m.get('reconciled_games',[]),
        unresolved_game_dates=pending_dates,source_manifest_sha256=sha(src/'manifest.json'),
        minute_feature_policy='nearest integer minute, half-up; rounded-zero excluded from personal model history',
        files={p.name:sha(p) for p in sorted(staging.iterdir())},
        models={v:dict(path=p,sha256=sha(ROOT/p),trained_through='2026-01-31') for v,p in MODEL_PATHS.items()})
    (staging/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    load_future(ROOT,staging)
    # Same-volume directory rename; preserve the old package and roll back on failure.
    archive=None;workspace_outputs=(ROOT/'outputs').resolve()
    if out.exists():
        archive=ROOT/'outputs/future_prediction_archives'/(sha(out/'manifest.json')[:16]+'_'+staging.name)
        archive.parent.mkdir(exist_ok=True)
        if not out.resolve().is_relative_to(workspace_outputs) or not archive.resolve().is_relative_to(workspace_outputs):raise ValueError('Archive path escaped outputs')
        out.replace(archive)
    try:
        if not staging.resolve().is_relative_to(workspace_outputs) or not out.resolve().is_relative_to(workspace_outputs):raise ValueError('Package path escaped outputs')
        staging.replace(out)
    except Exception:
        if archive is not None and not out.exists():archive.replace(out)
        raise
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True);p.add_argument('--replace',action='store_true')
    a=p.parse_args();build(a.snapshot,replace=a.replace)
