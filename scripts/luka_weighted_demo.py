"""User-requested reused-test demonstration. Frozen weights; never fit or tune."""
import json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.build_features_baseline import read_csv,save_csv,sha
from scripts.recent_game_features import canonical_date,season_for,number
from scripts.merge_game_rosters import matchup_info,roster_index,ALIASES
from scripts.same_conference_h2h import h2h_lineup_features
from scripts.high_score_weighting import VARIANTS,input_matrix,point_group_scores,alert_metrics
from scripts.experiment_two_stage import save_rows
from scripts.time_validation_tuning import dump
from nba_dashboard.data import expected_hash

ROOT=Path(__file__).resolve().parents[1]
PLAYER='Luka Dončić'


def build_prior_team_games(raw_rows,rosters,target_day):
    """Dedicated authorized demo path; original development cutoff remains intact."""
    canonical_date(target_day); season=season_for(target_day)
    groups={}; seen=set()
    for r in raw_rows:
        day=r['GAMEDATE']
        # Cut the date before accessing PTS/MIN, including the target game's values.
        if day>=target_day or season_for(day)!=season: continue
        canonical_date(day); team=r['TEAM']; pair,_=matchup_info(r['MATCHUP'],team)
        opponent=next(t for t in pair if t!=team); player=ALIASES.get(r['PLAYER'],r['PLAYER'])
        minute=number(r['MIN']); points=number(r['PTS'],integer=True)
        if (player,day) in seen or minute is None or points is None or (minute==0 and points!=0):
            raise ValueError('Duplicate or invalid prior outcome')
        seen.add((player,day)); key=day,team
        g=groups.setdefault(key,dict(opponent=opponent,points=0,appeared=set(),zero=set()))
        if g['opponent']!=opponent: raise ValueError('Ambiguous team/date')
        if minute>0: g['points']+=points; g['appeared'].add(player)
        else: g['zero'].add(player)
    result=[]
    for (day,team),g in sorted(groups.items()):
        other=groups.get((day,g['opponent']))
        if other is None or other['opponent']!=team or not g['appeared'] or not other['appeared']:
            raise ValueError('Missing side or appeared players')
        if rosters is not None:
            roster=rosters.get((day,team))
            if roster is None or roster['pair']!=frozenset([team,g['opponent']]) or g['appeared']-set(roster['players']) or set(roster['players'])-g['appeared']-g['zero']:
                raise ValueError('Incomplete prior source/roster coverage')
        result.append(dict(team_id=team,opponent_id=g['opponent'],GAMEDATE=day,
            game_id='|'.join([day,*sorted([team,g['opponent']])]),status='completed',
            points_for=g['points'],points_allowed=other['points'],combined_points=g['points']+other['points']))
    if rosters is not None:
        expected={key for key in rosters if key[0]<target_day and season_for(key[0])==season}
        if expected!=set(groups): raise ValueError('Prior roster game coverage mismatch')
    return result


def main():
    out=ROOT/'outputs/demos/luka_combined_w2_march_2026'
    if out.exists(): raise ValueError('Existing demonstration cannot be overwritten')
    source=ROOT/'outputs/week2/high_score_weighting_v7'
    modelpath=source/'models/February_combined_w2.joblib'
    rawpath=ROOT/'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv'
    rosterpath=rawpath.with_name('nba_2025_26_regular_season_game_rosters.csv')
    testpath=ROOT/'outputs/week1/team_lineup_features/test.csv'
    originalpath=ROOT/'outputs/week2/final_test/predictions_test.csv'
    protected=[p for p in (ROOT/'outputs/week2').rglob('*') if p.is_file()]
    protected += [rawpath,rosterpath,testpath,ROOT/'docs/week1/conferences_2025_26.json']
    before={str(p.relative_to(ROOT)):sha(p) for p in set(protected)}
    meta=json.loads((source/'validation.json').read_text(encoding='utf-8'))
    if sha(modelpath)!=expected_hash(meta['output_sha256'],modelpath.relative_to(source)):
        raise ValueError('Research model changed')
    bundle=joblib.load(modelpath)
    if bundle['variant']!='combined_w2' or bundle['weight_multiplier']!=2 or bundle['columns']!=VARIANTS['combined_w2']['columns'] or bundle['trained_through']!='2026-01-31':
        raise ValueError('Unexpected model/cutoff/feature order')
    raw=read_csv(rawpath); rosters=roster_index(read_csv(rosterpath))
    targets=sorted([r for r in read_csv(testpath) if r['player_id']==PLAYER and '2026-03-01'<=r['GAMEDATE']<='2026-04-12'],key=lambda r:r['GAMEDATE'])
    if not targets: raise ValueError('No eligible Luka games')
    raw_index={(r['PLAYER'],r['MATCHUP']+'|'+r['GAMEDATE']):r for r in raw}
    positive_keys={(r['PLAYER'],r['MATCHUP']+'|'+r['GAMEDATE']) for r in raw if r['PLAYER']==PLAYER and '2026-03-01'<=r['GAMEDATE']<='2026-04-12' and float(r['MIN'])>0}
    if positive_keys!={(r['player_id'],r['game_id']) for r in targets}: raise ValueError('Raw eligible game coverage differs')
    originals={(r['player_id'],r['game_id']):r for r in read_csv(originalpath)}
    out.mkdir(parents=True)
    # Explicit new user authorization: a demonstration on already-consumed test dates.
    dump(out/'demo_protocol.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),player=PLAYER,
        requested_start='2026-03-01',data_end='2026-04-12',eligible_n=len(targets),
        scope='explicit_user_requested_demonstration_on_previously_consumed_test_period',
        independent_test=False,fit_allowed=False,model_file=str(modelpath.relative_to(ROOT)),model_sha256=sha(modelpath),
        trained_through='2026-01-31',weight_multiplier=2,model_selection_allowed=False,
        prior_experiment_protocols_unchanged=True,historical_lineup='postgame_snapshot',date_isolation='strictly earlier dates'))
    features=[]
    for r in targets:
        day=r['GAMEDATE']; matchup,_=r['game_id'].rsplit('|',1); team=matchup.split()[0]
        teams=build_prior_team_games(raw,rosters,day)
        personal=[]
        for h in raw:
            if h['PLAYER']!=PLAYER or h['GAMEDATE']>=day or season_for(h['GAMEDATE'])!=season_for(day): continue
            personal.append(dict(player_id=PLAYER,team_id=h['TEAM'],game_id=h['MATCHUP']+'|'+h['GAMEDATE'],
                                 GAMEDATE=h['GAMEDATE'],**{c:h[c] for c in ['PTS','MIN','REB','AST']}))
        target={k:v for k,v in r.items() if not k.startswith('target_')}; target['team_id']=team
        starters=rosters[(day,team)]['starters']
        f=h2h_lineup_features(personal,teams,target,starters,mode='historical_postgame')
        if not f['eligible'] or f['live_prediction_ready']: raise ValueError('Invalid historical eligibility claim')
        for col in VARIANTS['basic_w2']['columns']:
            if not np.isclose(float(r[col]),float(f[col]),rtol=0,atol=1e-12): raise ValueError('Rebuilt base feature mismatch: '+col)
        features.append(dict(target,**f))
    save_rows(out/'features.csv',features)
    with threadpool_limits(limits=2):
        pred=bundle['model'].predict(input_matrix(features,'combined_w2')).astype(float)
        fake=[dict(r,target_pts=99999,PTS=99999,MIN=99999) for r in features]
        if not np.isfinite(pred).all() or not np.array_equal(pred,bundle['model'].predict(input_matrix(fake,'combined_w2')).astype(float)):
            raise ValueError('Invalid/outcome-dependent predictions')
        if not np.array_equal(pred,joblib.load(modelpath)['model'].predict(input_matrix(features,'combined_w2')).astype(float)):
            raise ValueError('Model reload mismatch')
    predictions=[]
    for r,f,p in zip(targets,features,pred):
        key=r['player_id'],r['game_id']; actual=raw_index[key]; original=originals[key]
        if float(actual['PTS'])!=float(r['target_pts']) or float(original['target_pts'])!=float(actual['PTS']): raise ValueError('Source outcome mismatch')
        y=float(actual['PTS']); old=float(original['pred_xgboost_starter'])
        predictions.append(dict(player_id=PLAYER,game_id=r['game_id'],GAMEDATE=r['GAMEDATE'],MATCHUP=actual['MATCHUP'],
            actual_minutes=float(actual['MIN']),actual_pts=y,pred_weighted=float(p),pred_original=old,
            pred_baseline=float(r['pts_mean_5']),weighted_abs_error=abs(y-p),original_abs_error=abs(y-old),
            weighted_residual=y-p,is_starter=int(float(f['is_starter'])),predicted_30_plus=bool(p>=30),actual_30_plus=bool(y>=30)))
    save_rows(out/'predictions.csv',predictions)
    human=[dict(比賽日期=r['GAMEDATE'],對戰=r['MATCHUP'],實際上場時間_分鐘=r['actual_minutes'],
                實際得分=r['actual_pts'],新模型預測=round(r['pred_weighted'],2),原模型預測=round(r['pred_original'],2),
                五場均值=round(r['pred_baseline'],2),新模型誤差=round(r['weighted_abs_error'],2),原模型誤差=round(r['original_abs_error'],2)) for r in predictions]
    save_rows(out/'Luka_3月起預測.csv',human)
    scores=[]; alerts=[]
    for variant,column in [('weighted','pred_weighted'),('original','pred_original'),('baseline','pred_baseline')]:
        y=[r['actual_pts'] for r in predictions]; values=[r[column] for r in predictions]
        scores.extend(dict(variant=variant,**s) for s in point_group_scores(y,values))
        alerts.append(dict(variant=variant,**alert_metrics(y,values)))
    save_rows(out/'scores.csv',scores); save_rows(out/'alert_metrics.csv',alerts)
    make_report(out,predictions,scores,alerts)
    if any(sha(ROOT/p)!=h for p,h in before.items()): raise ValueError('Original source/model/research changed')
    dump(out/'validation.json',dict(player=PLAYER,n=len(predictions),fits=0,independent_test=False,
        base9_rebuilt_equal=True,all_eligible_raw_games_included=True,model_reload_equal=True,
        target_outcomes_not_in_features=True,source_sha256=before,
        code_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
        output_sha256={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(output=str(out),n=len(predictions),scores=[s for s in scores if s['group'] in ['all','30+']],alerts=alerts),ensure_ascii=False))


def make_report(out,rows,scores,alerts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=dict(weighted='20項＋高分2倍',original='原九項模型',baseline='前五場均值')
    lines=['# Luka Dončić：2026年3月起加權模型展示','',
        f"範圍：{rows[0]['GAMEDATE']}～{rows[-1]['GAMEDATE']}，全部{len(rows)}場合格正分鐘出賽。模型只訓練到2026-01-31，固定使用v7的February_combined_w2，沒有重訓或選模。",
        '本次依使用者明確要求，在已使用過的三月起資料另作探索性展示；不是新的獨立最終測試。原研究協議、正式測試與線上模型均不改。',
        '每場特徵先切出嚴格前日資料再計算；包含先前已完成的三月賽事。歷史先發仍來自賽後名單，以實際出賽為前提，屬日期隔離的假設性回測。','',
        '|分組|模型|場數|MAE|RMSE|','|---|---|---:|---:|---:|']
    for r in scores:
        if r['group'] in ['all','30+']:
            lines.append(f"|{r['group']}|{labels[r['variant']]}|{r['n']}|{r['mae']:.2f}|{r['rmse']:.2f}|")
    lines+=['','## 逐場結果','', '|日期|對戰|分鐘|實際|新模型|原模型|新誤差|原誤差|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['GAMEDATE']}|{r['MATCHUP']}|{r['actual_minutes']:.0f}|{r['actual_pts']:.0f}|{r['pred_weighted']:.2f}|{r['pred_original']:.2f}|{r['weighted_abs_error']:.2f}|{r['original_abs_error']:.2f}|")
    lines+=['','## 高分警示','', '固定預測>=30才警示，不是高分機率。']
    for a in alerts:
        precision='未定義（無警示）' if a['precision'] is None else f"{a['precision']:.1%}"
        recall='未定義' if a['recall'] is None else f"{a['recall']:.1%}"
        lines.append(f"- {labels[a['variant']]}：抓到{a['tp']}場、誤報{a['fp']}場、漏掉{a['fn']}場；Precision {precision}、Recall {recall}。")
    lines+=['','同一球員的小樣本不能代表全聯盟或新球季表現；不據此更換權重、加特徵或宣稱正式升級。圖表及CSV另存同目錄，指標用完整精度計算。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(12,5))
    x=np.arange(len(rows))
    for col,label,color in [('actual_pts','Actual','#243C60'),('pred_weighted','20 features + 2x weights','#C85D32'),('pred_original','Original 9 features','#8794A5')]:
        ax.plot(x,[r[col] for r in rows],marker='o',label=label,color=color)
    ax.set(xticks=x,xticklabels=[r['GAMEDATE'][5:] for r in rows],ylabel='Points',title='Luka Doncic | March onward, 2026 | reused-data demonstration',ylim=(0,max(r['actual_pts'] for r in rows)+5))
    ax.tick_params(axis='x',rotation=45); ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(out/'luka_predictions.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    main()
