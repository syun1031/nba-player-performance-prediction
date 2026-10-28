"""Build basic features and a development-only last-five scoring baseline."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

if __package__:
    from .recent_game_features import FEATURE_COLUMNS, SPEC_VERSION, build_feature_rows
else:
    from recent_game_features import FEATURE_COLUMNS, SPEC_VERSION, build_feature_rows

ROOT=Path(__file__).resolve().parents[1]


def regression_metrics(actual,predicted):
    if len(actual)!=len(predicted):
        raise ValueError('Actual and prediction lengths differ')
    if not all(math.isfinite(float(x)) for x in actual+predicted):
        raise ValueError('Non-finite evaluation values')
    errors=[float(p)-float(y) for y,p in zip(actual,predicted)]
    return dict(n=len(errors),mae=sum(abs(e) for e in errors)/len(errors) if errors else None,
                rmse=math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None)


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\r\n')
        w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(row[k],ensure_ascii=False) if isinstance(row[k],list) else row[k] for k in fields})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/week1/features_baseline')
    args=parser.parse_args()
    path=ROOT/'outputs/week1/model_data/nba_2025_26_player_games_merged.csv'
    sources=[path,ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv',
             ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_game_rosters.csv']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    raw=read_csv(path)
    rows=build_feature_rows(raw)
    assert len(rows)==len(raw)==26568
    eligible=[r for r in rows if r['eligible']]
    assert all(all(v is not None and math.isfinite(v) for v in [r[c] for c in FEATURE_COLUMNS]) for r in eligible)
    assert all(r['history_count']>=5 and 5<=r['n_10']<=10 for r in eligible)
    assert all(r['last_history_date']<r['GAMEDATE'] for r in rows if r['last_history_date'])
    keys=[(r['player_id'],r['game_id']) for r in rows]
    assert len(set(keys))==len(rows)
    assert len(rows)==len(eligible)+sum(not r['eligible'] for r in rows)
    days={s:{r['GAMEDATE'] for r in rows if r['split']==s} for s in ('train','validation','test')}
    assert max(days['train'])<min(days['validation']) and max(days['validation'])<min(days['test'])

    # Verify order invariance and that changing later scores cannot affect earlier rows.
    assert build_feature_rows(list(reversed(raw)))==rows
    changed=[dict(r,PTS='999999') if r['GAMEDATE']>='2026-02-15' else dict(r) for r in raw]
    recomputed=build_feature_rows(changed)
    assert [r for r in recomputed if r['GAMEDATE']<'2026-02-15']==[r for r in rows if r['GAMEDATE']<'2026-02-15']

    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    feature_fields=list(rows[0])
    save_csv(out/'player_game_features.csv',rows,feature_fields)
    manifest_fields=['player_id','game_id','GAMEDATE','split','eligible','exclusion_reason','history_count','spec_version']
    save_csv(out/'common_sample_manifest.csv',rows,manifest_fields)
    model_fields=['player_id','game_id','GAMEDATE','split']+FEATURE_COLUMNS+['target_pts']
    split_summary={}
    predictions=[]
    for split in ('train','validation','test'):
        all_split=[r for r in rows if r['split']==split]
        sample=[r for r in eligible if r['split']==split]
        save_csv(out/(split+'.csv'),sample,model_fields)
        split_summary[split]=dict(source_rows=len(all_split),eligible_rows=len(sample),
            excluded_rows=len(all_split)-len(sample),coverage=len(sample)/len(all_split),
            players=len({r['player_id'] for r in sample}),
            first_date=min(r['GAMEDATE'] for r in all_split),last_date=max(r['GAMEDATE'] for r in all_split),
            exclusion_counts=dict(Counter(r['exclusion_reason'].split(';')[0] for r in all_split if not r['eligible'])))
        if split=='test':
            split_summary[split]['metrics_status']='not_evaluated'
            continue
        m=regression_metrics([r['target_pts'] for r in sample],[r['pts_mean_5'] for r in sample])
        split_summary[split]['baseline']=m
        for r in sample:
            predictions.append(dict(player_id=r['player_id'],game_id=r['game_id'],GAMEDATE=r['GAMEDATE'],
                split=split,actual_pts=r['target_pts'],predicted_pts=r['pts_mean_5'],
                error=r['pts_mean_5']-r['target_pts']))
    save_csv(out/'baseline_predictions_development.csv',predictions,
             ['player_id','game_id','GAMEDATE','split','actual_pts','predicted_pts','error'])

    saved=read_csv(out/'player_game_features.csv')
    assert len(saved)==len(rows)
    for original,disk in zip(rows,saved):
        assert original['game_id']==disk['game_id'] and original['player_id']==disk['player_id']
        assert json.loads(disk['history_game_ids_5'])==original['history_game_ids_5']
        for col in FEATURE_COLUMNS:
            assert (float(disk[col]) if disk[col] else None)==original[col]
    for split in ('train','validation'):
        disk=read_csv(out/(split+'.csv'))
        m=regression_metrics([float(r['target_pts']) for r in disk],[float(r['pts_mean_5']) for r in disk])
        assert m==split_summary[split]['baseline']
    assert all(p['split']!='test' for p in predictions)
    assert all(sha(p)==hashes[str(p.relative_to(ROOT))] for p in sources)
    summary=dict(verified=True,spec_version=SPEC_VERSION,feature_columns=FEATURE_COLUMNS,
                 source_sha256=hashes,source_rows=len(raw),eligible_rows=len(eligible),
                 excluded_rows=len(rows)-len(eligible),splits=split_summary,
                 source_files_unchanged=True,output_readback_verified=True,
                 actual_data_order_invariance=True,future_score_mutation_check=True,
                 test_metrics_computed=False,
                 limitations=['Historical post-season snapshot; prior-day information assumed available',
                              'No current-game lineup/absence features in this basic dataset',
                              'Appearance is conditional; does not predict whether a player plays'])
    (out/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report='''# 最近出賽特徵與五場平均基準

這一步建立八項基礎特徵，讓模型只看到本場以前的資訊；先用最近五場平均得分，作為之後三種模型要比較的起點。

## 結果

| 集合 | 正分鐘紀錄 | 可評估 | 覆蓋率 | MAE（分） | RMSE（分） |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
    for split,label in [('train','訓練'),('validation','驗證'),('test','測試')]:
        s=split_summary[split]
        m=s.get('baseline')
        mae=f"{m['mae']:.3f}" if m else '尚未評估'
        rmse=f"{m['rmse']:.3f}" if m else '尚未評估'
        report+=f"| {label} | {s['source_rows']:,} | {s['eligible_rows']:,} | {s['coverage']:.1%} | {mae} | {rmse} |\n"
    report+='''
MAE 是平均差幾分；RMSE 對較大的誤差更敏感。訓練集合的基準分數只是歷史區間表現，五場平均沒有擬合模型參數。沒有保證機器學習一定比較準。

## 規則

- 每筆只用同球員、同球季、前一天或更早的資料。跨隊保留歷史，跨季重置。
- 零分鐘不進個人窗口，有上場的零分保留。至少五場才預測；十場均值可以用五至九場的實際筆數。
- 八欄為 pts_mean_5、pts_mean_10、min_mean_5、reb_mean_5、ast_mean_5、is_home、is_neutral_site、rest_days。rest_days 是距上次個人出賽的中間天數，不是球隊休息或健康狀態。
- 每個樣本保留歷史窗口的 game_id 供核對；資料不足有原因，不補零或跳過壞資料。
- 依已確認日期切分，二月驗證、三月起測試。較早驗證／測試比賽可作較晚日期的歷史，但不因此重訓模型。
- 測試只輸出資料與資格清單，不計算誤差或輸出基準預測。

## 檔案

- player_game_features.csv：全部正分鐘樣本、八項特徵、歷史窗口及資格原因。
- common_sample_manifest.csv：所有樣本的鍵、集合與資格；之後各模型共用，不可各自刪列後比數字。
- train.csv、validation.csv、test.csv：合格樣本與目標值。訓練只能選 validation.json 的 feature_columns，不能把 target_pts、身份或其他稽核欄位當輸入。
- baseline_predictions_development.csv：僅訓練／驗證的預測與誤差。
- validation.json：完整指標、覆蓋率、來源雜湊與驗證結果。

## 驗證與限制

已檢查來源不變、輸出重新讀取一致、切分日期分開、窗口日期嚴格早於本場、整份資料倒序不影響結果，以及改動二月十五日以後得分不影響此前結果。

這仍是球季結束後資料的假設性回測，無法證明每筆資料當時何時公布。本場先發與缺席未加入這份基礎資料；陣容擴充需另做，不能把賽後未上場直接當賽前確認缺席。未來賽程／名單來源也尚未完成。

## 重現

從專案根目錄執行 `python scripts/build_features_baseline.py`；測試使用 `python -m unittest discover -s tests`。只需 Python 標準函式庫。
'''
    (out/'README.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
