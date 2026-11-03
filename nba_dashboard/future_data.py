"""Small verified current-season package, separate from frozen backtests."""
from pathlib import Path
from datetime import date,timedelta
import csv,hashlib,json,math
from .data import DataIssue
from .future_source import aware
from scripts.fetch_current_season import played_seconds

MODEL_PATHS={
    'combined_prior25_w2':'outputs/week2/prior_season_profile_v9/models/February_combined_prior25_w2.joblib',
    'residual25_w2':'outputs/offset_prior25_w2_2025_26/models/February_residual25_w2.joblib',
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history_issues(bundle,day,now):
    reasons=[];m=bundle['manifest']
    try:
        if date.fromisoformat(m['through'])<date.fromisoformat(day)-timedelta(days=1):reasons.append('history_not_updated')
        if aware(m['collected_at_utc'])>now:reasons.append('history_not_available')
        if any(r['GAMEDATE']<day for r in bundle['pending']):reasons.append('unresolved_prior_games')
    except (KeyError,ValueError,TypeError):reasons.append('invalid_history_package')
    return reasons


def model_minutes(row):
    seconds=played_seconds(row['minutes_source'])
    if seconds is None or seconds<=0:raise DataIssue('正分鐘歷史缺少原始分秒。')
    if not math.isclose(seconds/60,float(row['MIN']),abs_tol=1e-9):raise DataIssue('上場分鐘與原始分秒不一致。')
    return int((seconds+30)//60)


def load_future(root,folder=None):
    root=Path(root);folder=Path(folder) if folder is not None else root/'outputs/future_prediction'
    try:
        m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if m['schema']!=1 or m['season']!='2026-27' or m['models'].keys()!=MODEL_PATHS.keys():raise DataIssue('未來預測資料版本不相符。')
        def read(name):
            path=folder/name
            if sha(path)!=m['files'][name]:raise DataIssue('未來資料檔案已變動：'+name)
            if name.endswith('.json'):return json.loads(path.read_text(encoding='utf-8'))
            with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
        players=read('player_games.csv');teams=read('team_games.csv');pending=read('pending_schedule.csv')
        profiles=read('previous_season_profiles.csv');conferences=read('conferences_2026_27.json')
        if 'status_resolution.json' in m['files']:read('status_resolution.json')
        if conferences['season']!='2026-27' or len(conferences['teams'])!=30 or any(list(conferences['teams'].values()).count(c)!=15 for c in ['East','West']):raise DataIssue('本季分區對照不完整。')
        ids={};names={};keys=set();model_history=[]
        for r in players:
            day=r['GAMEDATE'];date.fromisoformat(day);pid=r['nba_player_id'];name=r['player_id']
            if r['season']!='2026-27' or not '2026-07-01'<=day<=m['through'] or not pid.isdigit() or r['game_id']!=r['MATCHUP']+'|'+day or r['PLAYER']!=name or r['TEAM']!=r['team_id']:
                raise DataIssue('本季球員身份或日期不一致。')
            if pid in ids and ids[pid]!=name or name in names and names[name]!=pid:raise DataIssue('球員姓名與官方ID對照有衝突。')
            ids[pid]=name;names[name]=pid
            key=pid,day
            if key in keys:raise DataIssue('本季球員日期重複。')
            keys.add(key)
            for field in ['MIN','PTS','REB','AST']:
                r[field]=float(r[field])
                if not math.isfinite(r[field]) or r[field]<0 or field!='MIN' and not r[field].is_integer():raise DataIssue('本季球員數據無效。')
            if r['MIN']<=0:raise DataIssue('本季個人歷史含零分鐘。')
            minute=model_minutes(r)
            if minute>0:model_history.append(dict(r,MIN=minute))
        teamkeys=set()
        for r in teams:
            key=r['team_id'],r['GAMEDATE']
            if key in teamkeys or r['status']!='completed' or r['team_id'] not in conferences['teams'] or not '2026-07-01'<=r['GAMEDATE']<=m['through']:
                raise DataIssue('球隊歷史日期或狀態不完整。')
            teamkeys.add(key)
            for field in ['points_for','points_allowed','combined_points']:
                r[field]=int(r[field])
                if r[field]<0:raise DataIssue('球隊比分無效。')
            if r['combined_points']!=r['points_for']+r['points_allowed']:raise DataIssue('球隊比分加總不符。')
            selected=[p for p in players if p['nba_game_id']==r['nba_game_id'] and p['team_id']==r['team_id']]
            if not selected or sum(p['PTS'] for p in selected)!=r['points_for']:raise DataIssue('球員與球隊分數無法核對。')
        if len(players)!=m['player_rows'] or len(teams)!=m['team_rows']:raise DataIssue('資料筆數不相符。')
        profile_map={}
        for r in profiles:
            pid=r['nba_player_id']
            if pid in profile_map or not pid.isdigit():raise DataIssue('上季球員身份重複。')
            numbers={k:float(v) for k,v in r.items() if k.startswith('prev_season_') or k=='has_prev_season'}
            if not all(math.isfinite(v) and v>=0 for v in numbers.values()) or numbers['prev_season_games']<1 or numbers['has_prev_season']!=1:raise DataIssue('上季表現無效。')
            profile_map[pid]=numbers
        for variant,path in MODEL_PATHS.items():
            info=m['models'][variant]
            if info['path']!=path or sha(root/path)!=info['sha256']:raise DataIssue('固定模型檔案無法驗證。')
        return dict(history=model_history,raw_history=players,team_games=teams,profiles=profile_map,conferences=conferences,
                    pending=pending,manifest=m,id_names=ids)
    except DataIssue:raise
    except (OSError,ValueError,KeyError,TypeError,OverflowError) as exc:raise DataIssue('未來預測資料尚未備齊或格式不正確。') from exc


HISTORY_REASONS={'history_not_updated':'歷史尚未涵蓋目標日期的前一天，請先手動更新資料。',
    'history_not_available':'這份歷史資料是在目前時間之後取得，不能使用。',
    'unresolved_prior_games':'較早日期仍有官網狀態未釐清的比賽，請先更新並核對歷史。',
    'invalid_history_package':'歷史資料未通過驗證。'}
