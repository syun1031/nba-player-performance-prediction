"""Encode eligible historical starting teammates using training names only."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from shutil import copyfile

if __package__:
    from .teammate_encoder import TeammateEncoder,validate_names
    from .build_features_baseline import read_csv,save_csv,sha
else:
    from teammate_encoder import TeammateEncoder,validate_names
    from build_features_baseline import read_csv,save_csv,sha

ROOT=Path(__file__).resolve().parents[1]
VERSION='teammate-multihot-v1'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/week1/teammate_encoding')
    args=parser.parse_args()
    source=ROOT/'outputs/week1/team_lineup_features'
    protected=[p for p in source.iterdir() if p.is_file()]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    original=read_csv(source/'player_game_features.csv')
    rows=[r for r in original if r['eligible']=='True']
    if any(r['eligible'] not in ('True','False') for r in original):
        raise ValueError('Unknown sample eligibility')
    rows.sort(key=lambda r:(r['GAMEDATE'],r['player_id'],r['game_id']))
    groups={s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
    assert sum(map(len,groups.values()))==len(rows)==23722
    base_features=json.loads((source/'validation.json').read_text(encoding='utf-8'))['numeric_feature_columns']
    assert len(base_features)==9 and 'target_pts' not in base_features
    teammate_lists={}
    for r in rows:
        names=validate_names(json.loads(r['starting_teammates']))
        key=r['player_id'],r['game_id']
        if key in teammate_lists:
            raise ValueError('Duplicate sample key')
        if r['player_id'] in names or len(names)!=(4 if int(r['is_starter']) else 5):
            raise ValueError('Own player must not be encoded as a teammate; roster size mismatch')
        teammate_lists[key]=names
        assert all(math.isfinite(float(r[c])) for c in base_features+['target_pts'])
    train_names=[teammate_lists[(r['player_id'],r['game_id'])] for r in groups['train']]
    encoder=TeammateEncoder().fit(train_names)
    expected=sorted({n for names in train_names for n in names})
    assert encoder.classes_==expected
    assert TeammateEncoder().fit(list(reversed(train_names))).to_dict()==encoder.to_dict()
    state=encoder.to_dict()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    (out/'teammate_encoder.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    restored=TeammateEncoder.from_dict(json.loads((out/'teammate_encoder.json').read_text(encoding='utf-8')))
    encoded_features=encoder.get_feature_names_out()
    numeric_features=base_features+encoded_features
    assert len(set(numeric_features))==len(numeric_features)
    header=['player_id','game_id','GAMEDATE','split']+numeric_features+['target_pts','lineup_source_kind','pregame_verified']
    mapping=[dict(feature_name=field,teammate_name=name,meaning='1=此人在本場先發隊友名單，0=不在')
             for field,name in zip(encoded_features[:-1],encoder.classes_)]
    mapping.append(dict(feature_name='teammate_unknown_count',teammate_name='',meaning='訓練集未見隊友人數，不新增類別'))
    save_csv(out/'feature_dictionary.csv',mapping,['feature_name','teammate_name','meaning'])
    copyfile(source/'common_sample_manifest.csv',out/'common_sample_manifest.csv')
    split_summary={};unseen_rows=[]
    for split,samples in groups.items():
        unseen=Counter();unknown_samples=0;max_unknown=0
        with (out/(split+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=header,lineterminator='\r\n')
            w.writeheader()
            for r in samples:
                names=teammate_lists[(r['player_id'],r['game_id'])]
                values=encoder.transform([names])[0]
                assert values==restored.transform([list(reversed(names))])[0]
                assert all(v in (0,1) for v in values[:-1])
                assert sum(values)==len(names) and values[-1]==sum(n not in encoder._index for n in names)
                if r['player_id'] in encoder._index:
                    assert values[encoder._index[r['player_id']]]==0
                unseen.update(n for n in names if n not in encoder._index)
                unknown_samples+=int(values[-1]>0);max_unknown=max(max_unknown,values[-1])
                result={k:r[k] for k in ['player_id','game_id','GAMEDATE','split']+base_features+
                        ['target_pts','lineup_source_kind','pregame_verified']}
                result.update(zip(encoded_features,values))
                w.writerow(result)
        split_summary[split]=dict(rows=len(samples),unknown_teammate_samples=unknown_samples,
            unknown_sample_rate=unknown_samples/len(samples) if samples else None,
            unknown_teammate_occurrences=sum(unseen.values()),distinct_unknown_names=len(unseen),
            max_unknown_in_sample=max_unknown)
        unseen_rows.extend(dict(split=split,teammate_name=name,occurrences=count) for name,count in sorted(unseen.items()))
        # Independently read the output and verify every dictionary flag against the input list.
        with (out/(split+'.csv')).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            assert reader.fieldnames==header
            count=0
            for r,saved in zip(samples,reader,strict=True):
                names=set(teammate_lists[(r['player_id'],r['game_id'])])
                assert (saved['player_id'],saved['game_id'])==(r['player_id'],r['game_id'])
                for field,name in zip(encoded_features[:-1],encoder.classes_):
                    assert int(saved[field])==int(name in names)
                assert int(saved['teammate_unknown_count'])==len(names-set(encoder.classes_))
                assert all(saved[c]==r[c] for c in base_features+['target_pts','pregame_verified','lineup_source_kind'])
                count+=1
            assert count==len(samples)
    assert split_summary['train']['unknown_teammate_samples']==0
    assert encoder.to_dict()==state
    save_csv(out/'unknown_teammates.csv',unseen_rows,['split','teammate_name','occurrences'])
    assert sha(out/'common_sample_manifest.csv')==sha(source/'common_sample_manifest.csv')
    assert all(sha(p)==hashes[str(p.relative_to(ROOT))] for p in protected)
    summary=dict(verified=True,spec_version=VERSION,source_spec_version='team-lineup-v1',
        fit_split='train',fit_samples=len(groups['train']),fit_end_date=max(r['GAMEDATE'] for r in groups['train']),
        vocabulary_size=len(encoder.classes_),base_numeric_columns=base_features,
        encoded_teammate_columns=encoded_features,numeric_feature_columns=numeric_features,
        total_numeric_columns=len(numeric_features),splits=split_summary,
        source_sha256=hashes,encoder_sha256=sha(out/'teammate_encoder.json'),
        source_files_unchanged=True,common_sample_manifest_unchanged=True,
        output_readback_verified=True,serialization_roundtrip_verified=True,
        training_order_invariance=True,no_vocabulary_growth_during_transform=True,
        labels_used_for_encoding=False,test_metrics_computed=False,
        historical_postgame_lineup=True,
        limitations=['Actual historical starting lineup, not verified pregame publication',
                     'Unknown count is constant zero in training and does not teach unseen-player effects',
                     'No teammate absence reference or trained prediction model',
                     'Scikit-learn integration is not yet executed; this step uses the Python standard library'])
    (out/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=f'''# 先發隊友類別編碼

更新日期：2026-10-22。已將先發隊友名單轉為 multi-hot 數值。每位訓練集隊友各一欄：本場在先發隊友名單中填 1，不在填 0；不把姓名編號當作有大小關係的數字，名單順序也不影響結果。

## 結果

只用 {len(groups['train']):,} 筆合格訓練樣本建立 {len(encoder.classes_)} 位隊友類別。加上未知人數欄與原九項數值，共 {len(numeric_features)} 個模型輸入。樣本資格與舊版完全相同，原始及舊版資料未修改。

| 集合 | 樣本數 | 含未知隊友的樣本 | 比例 |
| --- | ---: | ---: | ---: |
'''
    for split,label in [('train','訓練'),('validation','驗證'),('test','測試')]:
        s=split_summary[split]
        report+=f"| {label} | {s['rows']:,} | {s['unknown_teammate_samples']:,} | {s['unknown_sample_rate']:.1%} |\n"
    report+='''
## 未知隊友與時間規則

驗證、測試或未來資料遇到訓練未見姓名時，不改編碼器、不新增欄位，只增加 teammate_unknown_count。這能讓未知資訊被記錄，但訓練時未知人數必然為零，不能宣稱模型已學會未知球員的影響。不能因看見測試有未知隊友就回頭擴充詞彙。

編碼完全不使用得分標籤，也不使用驗證／測試名單建立類別。測試只核對欄位與未知覆蓋率，沒有訓練模型或計算測試分數。這份資料仍來自歷史賽後先發，pregame_verified=False。

## 檔案與使用

- train.csv、validation.csv、test.csv：數值模型資料；只使用 validation.json 的 numeric_feature_columns 作輸入。
- target_pts 是目標，player_id、game_id、日期、集合及來源旗標都是說明欄，不作數值特徵。
- teammate_encoder.json：可還原的訓練類別與固定欄位順序，必須與模型一起保存。
- feature_dictionary.csv：teammate_0001 等欄位與球員姓名的對照；0001 只是欄名，不是大小分數。
- common_sample_manifest.csv：沿用上一版的完整共同樣本清單。
- unknown_teammates.csv：各集合的未知姓名與出現次數，只用於追蹤覆蓋。
- validation.json：來源雜湊、欄位、樣本與驗證結果。

## 驗證與重現

已逐列重新讀檔，核對每個隊友旗標；已知旗標加未知人數等於四／五位隊友，自己不會被當成隊友。編碼器存檔還原、名單與訓練順序變更均不影響結果，來源與共同樣本清單未變。

執行 `python scripts/encode_teammates.py` 可重建；測試用 `python -m unittest discover -s tests`。此步只用標準函式庫，不需安裝 Scikit-learn；後續模型階段再驗證完整訓練流程。隊友缺席比較仍未實作。
'''
    (out/'README.md').write_text(report,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['verified','vocabulary_size','total_numeric_columns','splits',
                     'fit_end_date','source_files_unchanged','output_readback_verified']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
