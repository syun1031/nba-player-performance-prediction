"""Add team warmup and actual lineup columns to a separate historical dataset."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

if __package__:
    from .team_lineup_features import SPEC_VERSION,TeamHistoryIndex,lineup_for_request
    from .recent_game_features import FEATURE_COLUMNS
    from .merge_game_rosters import roster_index,matchup_info
    from .build_features_baseline import read_csv,save_csv,sha,regression_metrics
else:
    from team_lineup_features import SPEC_VERSION,TeamHistoryIndex,lineup_for_request
    from recent_game_features import FEATURE_COLUMNS
    from merge_game_rosters import roster_index,matchup_info
    from build_features_baseline import read_csv,save_csv,sha,regression_metrics

ROOT=Path(__file__).resolve().parents[1]
MODEL_FEATURES=FEATURE_COLUMNS+['is_starter']


def add_lineup_fields(base_rows,merged_rows,rosters):
    index=roster_index(rosters)
    team_games=[dict(team_id=team,GAMEDATE=day,game_id=f'{team}|{r["raw_matchup"]}|{day}',status='completed')
                for (day,team),r in index.items()]
    history=TeamHistoryIndex(team_games)
    source={(r['player_id'],r['game_id']):r for r in merged_rows}
    if len(source)!=len(merged_rows):
        raise ValueError('Duplicate merged sample keys')
    result=[];seen=set()
    for raw in sorted(base_rows,key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id'])):
        key=raw['player_id'],raw['game_id']
        if key in seen:
            raise ValueError('Duplicate base sample key')
        seen.add(key)
        current=source[key]
        roster=index.get((current['GAMEDATE'],current['team_id']))
        if current['GAMEDATE']!=raw['GAMEDATE'] or current['team_id']!=raw['team_id']:
            raise ValueError('Base/merged identity mismatch')
        if roster is None or roster['pair']!=matchup_info(current['MATCHUP'],current['TEAM'])[0]:
            raise ValueError('Missing or different current roster opponent')
        row=dict(raw)
        if raw['eligible'] not in ('True','False'):
            raise ValueError('Invalid base eligibility')
        basic_ok=raw['eligible']=='True'
        for c in FEATURE_COLUMNS+['target_pts']:
            row[c]=float(raw[c]) if raw[c] else None
        for c in ('history_game_ids_5','history_game_ids_10'):
            row[c]=json.loads(raw[c])
        for c in ('history_count','n_10'):
            row[c]=int(raw[c])
        lineup=lineup_for_request(history,current,roster['starters'],mode='historical_postgame')
        reasons=[r for r in [raw['exclusion_reason'],lineup['lineup_exclusion_reason']] if r]
        row.update(lineup,basic_eligible=basic_ok,eligible=basic_ok and lineup['lineup_eligible'],
                   exclusion_reason=';'.join(reasons),base_spec_version=raw['spec_version'],spec_version=SPEC_VERSION)
        if lineup['is_starter']!=int(current['is_starter']):
            raise ValueError('Current starter differs from merged source')
        if set(lineup['starting_teammates'])!=set(json.loads(current['starting_teammates'])):
            raise ValueError('Starting teammates differ from merged source')
        result.append(row)
    return result,team_games


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/week1/team_lineup_features')
    args=parser.parse_args()
    base=ROOT/'outputs/week1/features_baseline'
    merged=ROOT/'outputs/week1/model_data/nba_2025_26_player_games_merged.csv'
    roster=ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_game_rosters.csv'
    sources=[p for p in base.iterdir() if p.is_file()]+[merged,roster]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    base_rows,merged_rows,rosters=read_csv(base/'player_game_features.csv'),read_csv(merged),read_csv(roster)
    rows,team_games=add_lineup_fields(base_rows,merged_rows,rosters)
    assert len(rows)==len(base_rows)==26568
    assert Counter(g['team_id'] for g in team_games)==Counter({t:82 for t in {g['team_id'] for g in team_games}})
    assert len(team_games)==2460
    reversed_rows,_=add_lineup_fields(list(reversed(base_rows)),list(reversed(merged_rows)),list(reversed(rosters)))
    assert reversed_rows==rows
    assert all(not r['pregame_verified'] and not r['live_prediction_ready'] for r in rows)
    for r in rows:
        if r['team_history_last_date']:
            assert r['team_history_last_date']<r['GAMEDATE']
        assert len(r['starting_teammates'])==(4 if r['is_starter'] else 5)
        if r['eligible']:
            assert r['history_count']>=5 and r['team_history_count']>=5
            assert all(v is not None and math.isfinite(v) for v in (r[c] for c in MODEL_FEATURES))
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'player_game_features.csv',rows,list(rows[0]))
    manifest=['player_id','game_id','GAMEDATE','split','basic_eligible','lineup_eligible','eligible',
              'exclusion_reason','history_count','team_history_count','pregame_verified','spec_version']
    save_csv(out/'common_sample_manifest.csv',rows,manifest)
    split_summary={};predictions=[]
    for split in ('train','validation','test'):
        all_rows=[r for r in rows if r['split']==split]
        good=[r for r in all_rows if r['eligible']]
        save_csv(out/(split+'.csv'),good,['player_id','game_id','GAMEDATE','split']+MODEL_FEATURES+
                 ['target_pts','lineup_source_kind','pregame_verified'])
        s=dict(source_rows=len(all_rows),eligible_rows=len(good),
               coverage=len(good)/len(all_rows),
               exclusion_counts=dict(Counter(r['exclusion_reason'] for r in all_rows if not r['eligible'])))
        if split!='test':
            s['baseline']=regression_metrics([r['target_pts'] for r in good],[r['pts_mean_5'] for r in good])
            predictions.extend(dict(player_id=r['player_id'],game_id=r['game_id'],GAMEDATE=r['GAMEDATE'],
                split=split,actual_pts=r['target_pts'],predicted_pts=r['pts_mean_5']) for r in good)
        else:
            s['metrics_status']='not_evaluated'
        split_summary[split]=s
    save_csv(out/'baseline_predictions_development.csv',predictions,
             ['player_id','game_id','GAMEDATE','split','actual_pts','predicted_pts'])
    saved=read_csv(out/'player_game_features.csv')
    assert len(saved)==len(rows)
    for a,b in zip(rows,saved):
        assert (a['player_id'],a['game_id'])==(b['player_id'],b['game_id'])
        assert a['team_history_count']==int(b['team_history_count'])
        assert a['starting_teammates']==json.loads(b['starting_teammates'])
        assert a['confirmed_starting_players']==json.loads(b['confirmed_starting_players'])
        assert a['team_history_game_ids_5']==json.loads(b['team_history_game_ids_5'])
    for split in ('train','validation'):
        disk=read_csv(out/(split+'.csv'))
        assert regression_metrics([float(r['target_pts']) for r in disk],[float(r['pts_mean_5']) for r in disk])==split_summary[split]['baseline']
    assert all(sha(p)==hashes[str(p.relative_to(ROOT))] for p in sources)
    eligible=sum(r['eligible'] for r in rows)
    summary=dict(verified=True,spec_version=SPEC_VERSION,source_sha256=hashes,
        numeric_feature_columns=MODEL_FEATURES,teammate_identity_encoding='not_implemented',
        source_rows=len(rows),team_game_rows=len(team_games),team_count=30,eligible_rows=eligible,
        excluded_rows=len(rows)-eligible,
        team_history_insufficient_rows=sum(r['team_history_count']<5 for r in rows),
        newly_excluded_by_team_gate=sum(r['basic_eligible'] and r['team_history_count']<5 for r in rows),
        actual_starter_rows=sum(r['is_starter']==1 for r in rows),splits=split_summary,
        historical_postgame_lineup=True,live_prediction_ready_rows=0,
        source_files_and_base_outputs_unchanged=True,output_readback_verified=True,
        order_invariance=True,test_metrics_computed=False,
        limitations=['Actual historical roster, no verified pregame publication timestamps',
                     'No regular-starter absence reference or teammate encoding',
                     'Future source and end-to-end prediction workflow not implemented'])
    (out/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=f'''# 球隊五場門檻與本場先發欄位

更新日期：2026-10-22。原八項基礎資料保留不變，本目錄是另外產生的歷史陣容研究版本。

## 規則

- 球員與本場球隊各有至少五場先前紀錄才納入。球隊場數讀完整 2,460 筆球隊比賽紀錄，不從個人分鐘數推算。
- 僅計同季、前一天或更早已完成的球隊比賽；同日與未來不計。轉隊按目前球隊檢查，跨季重置。
- is_starter 直接由本場五人名單取得，不按分鐘排序。先發球員有四名先發隊友；替補球員有五名，替補仍可納入。
- 歷史資料使用星號代表的實際先發，lineup_source_kind=historical_postgame、pregame_verified=False。不能宣稱已核對當時何時公布，也不能直接拿本資料說未來服務已可用。
- 共用單筆入口的預設為賽前模式，需提供確認名單及帶時區的取得／預測／預定開賽時間。資料不足、取得晚於預測或已開賽都拒絕放行；這只是輸入檢查，不代表已連接新賽季來源。

## 筆數

全部 {len(rows):,} 筆正分鐘資料，其中 {eligible:,} 筆合格、{len(rows)-eligible:,} 筆排除；新球隊門檻另排除 {summary['newly_excluded_by_team_gate']:,} 筆原基礎模型合格樣本。

| 集合 | 合格樣本 | 五場平均 MAE | RMSE |
| --- | ---: | ---: | ---: |
'''
    for split,label in [('train','訓練'),('validation','驗證'),('test','測試')]:
        s=split_summary[split];m=s.get('baseline')
        report+=f"| {label} | {s['eligible_rows']:,} | {m['mae']:.3f} | {m['rmse']:.3f} |\n" if m else f"| {label} | {s['eligible_rows']:,} | 尚未評估 | 尚未評估 |\n"
    report+='''
## 輸出與使用

- player_game_features.csv：八項基礎特徵、is_starter、先發五人與先發隊友名單、個人與球隊歷史數及排除原因。
- common_sample_manifest.csv：各模型共用的樣本資格清單。
- train.csv、validation.csv、test.csv：可用樣本；數值模型只取 validation.json 的 numeric_feature_columns，不取 target_pts 或稽核欄位。
- baseline_predictions_development.csv：相同樣本上的訓練／驗證五場平均；測試不評分。
- 隊友姓名先以 JSON 陣列保存，順序不代表位置；類別編碼與缺席參考名單尚未實作。

缺席不是「沒先發」；本工作沒有推定缺席者或新增未上場零分樣本。九項數值輸入中含實際歷史 is_starter，需以假設性陣容研究呈現。完整來源與驗證結果見 validation.json。

## 重現

從專案根目錄執行 `python scripts/build_team_lineup_features.py`；測試執行 `python -m unittest discover -s tests`。只需標準函式庫。
'''
    (out/'README.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
