"""Independent saved-output audit; no fitting or test-period predictions."""
import json
import math
import platform
import importlib.metadata
from pathlib import Path
from collections import defaultdict

from scripts.build_features_baseline import read_csv,save_csv,sha
from scripts.experiment_recent_changes import ROOT,VARIANTS,group_scores
from scripts.time_validation_tuning import dump


def main():
    out=ROOT/'outputs/week2/recent_change_v2'
    manifest=json.loads((out/'validation.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in manifest['source_sha256'].items())
    # Scores/report can be refreshed from the exact saved predictions without a refit.
    assert all(sha(out/p)==h for p,h in manifest['output_sha256'].items())
    by_player=defaultdict(list)
    for r in read_csv(ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv'):
        if r['GAMEDATE']>'2026-02-28':
            continue
        if float(r['MIN'])>0:
            by_player[r['PLAYER']].append(r)
    checked=0
    for split in ['train','validation']:
        original=read_csv(ROOT/f'outputs/week1/team_lineup_features/{split}.csv')
        expected={(r['player_id'],r['game_id']):r for r in original}
        rows=read_csv(out/f'features/{split}.csv')
        assert len(rows)==len(expected)
        for r in rows:
            old=expected[(r['player_id'],r['game_id'])]
            assert all(r[k]==v for k,v in old.items())
            prior=sorted([h for h in by_player[r['player_id']] if '2025-07-01'<=h['GAMEDATE']<r['GAMEDATE']],key=lambda h:h['GAMEDATE'])
            assert len(prior)>=5
            window=prior[-5:]; minutes=[float(h['MIN']) for h in window]; pts=[float(h['PTS']) for h in window]
            mean_min=sum(minutes)/5; mean_pts=sum(pts)/5
            independent=[pts[-1],sum(minutes[-3:])/3,minutes[-1]-mean_min,
                         math.sqrt(sum((v-mean_pts)**2 for v in pts)/5),
                         math.sqrt(sum((v-mean_min)**2 for v in minutes)/5)]
            for key,value in zip(VARIANTS['v2'][-5:],independent):
                assert math.isclose(float(r[key]),value,rel_tol=0,abs_tol=1e-12),(r['player_id'],key)
            for n in [3,5,10]:
                assert json.loads(r[f'history_game_ids_{n}'])==[h['MATCHUP']+'|'+h['GAMEDATE'] for h in prior[-n:]]
            assert r['last_history_date']==prior[-1]['GAMEDATE']
            assert int(r['history_count'])==len(prior)
            checked+=1
    cv=read_csv(out/'predictions_CV_pooled.csv'); feb=read_csv(out/'predictions_February.csv')
    assert len(cv)==9023 and len(feb)==3462
    assert len({(r['player_id'],r['game_id']) for r in cv+feb})==len(cv)+len(feb)
    periods=[(name,[r for r in cv if r['period']==name]) for name in ['F1','F2','F3']]+[('February',feb),('CV_pooled',cv)]
    original_scores=read_csv(out/'scores.csv'); scores=[]
    for period,rows in periods:
        assert all(r['GAMEDATE']<='2026-02-28' for r in rows)
        y=[float(r['target_pts']) for r in rows]; baseline=[float(r['pred_baseline']) for r in rows]
        for variant in VARIANTS:
            pred=[float(r['pred_'+variant]) for r in rows]
            for score in group_scores(y,pred,baseline):
                previous=next(s for s in original_scores if s['period']==period and s['variant']==variant and s['group']==score['group'])
                for key in ['n','mae','rmse','baseline_mae','baseline_rmse']:
                    assert math.isclose(float(previous[key]),score[key],rel_tol=0,abs_tol=1e-12)
                scores.append(dict(period=period,variant=variant,**score))
    save_csv(out/'scores.csv',scores,list(scores[0]))
    report=(out/'report.md').read_text(encoding='utf-8')
    section=['','## 高得分場次與判讀','',
             '|期間／實際得分|筆數|九項 MAE|十四項 MAE|','|---|---:|---:|---:|']
    for period in ['CV_pooled','February']:
        for group in ['20+','30+']:
            a,b=[next(s for s in scores if s['period']==period and s['variant']==v and s['group']==group) for v in VARIANTS]
            section.append(f"|{period}／{group}|{a['n']}|{a['mae']:.4f}|{b['mae']:.4f}|")
    section+=['','殘差定義為實際減預測，正值代表低估。20+與30+有重疊，不能相加。',
              '二月整體MAE只降低約0.14%，三輪合併約0.17%；兩個日期重抽區間均包含0，不能排除抽樣波動。',
              '三輪與二月的30+ MAE皆略增；新增特徵尚未改善爆分場次。建議保留原九項正式模型，十四項只作研究。']
    if '## 高得分場次與判讀' not in report:
        (out/'report.md').write_text(report+'\n'.join(section)+'\n',encoding='utf-8')
    audit=dict(independent_feature_rows=checked,cohort_original_values_equal=True,
               manual_window_formulas_equal=True,saved_scores_recomputed=True,
               residual_definition='actual_minus_prediction',additional_fits=0,test_predictions_computed=False)
    audit['environment']=dict(python=platform.python_version(),packages={
        name:importlib.metadata.version(name) for name in ['numpy','scikit-learn','xgboost','joblib','matplotlib']})
    dump(out/'independent_audit.json',audit)
    manifest['independent_audit']=audit
    manifest['code_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [
        ROOT/'scripts/recent_change_features.py',ROOT/'scripts/experiment_recent_changes.py',Path(__file__).resolve()]}
    manifest['output_sha256']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='validation.json'}
    dump(out/'validation.json',manifest)
    print(json.dumps(audit))


if __name__=='__main__':
    main()
