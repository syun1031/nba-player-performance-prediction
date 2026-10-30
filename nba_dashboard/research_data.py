"""Self-contained verified display bundle; never loads models or makes predictions."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .data import DataIssue,validate_predictions

MODELS=['basic9','basic_prior14','combined_prior25_w2']
NAMES={'baseline':'五場均值基準','basic9':'原九項','basic_prior14':'原九項＋上季表現',
       'combined_prior25_w2':'1＋3＋同區＋上季表現＋2倍'}
EXPECTED={'2025-26':{'CV':('2025-12-01','2026-01-31',9023),'validation':('2026-02-01','2026-02-28',3462),'test':('2026-03-01','2026-04-12',6979)},
          '2024-25':{'CV':('2024-12-01','2025-01-31',8647),'validation':('2025-02-01','2025-02-28',3634),'test':('2025-03-01','2025-04-13',7060)}}


@dataclass
class ResearchBundle:
    predictions:dict
    manifest:dict


def research_metrics(frame,models,threshold=None):
    if len(set(models))!=len(models) or any(m not in ['baseline']+MODELS for m in models):raise DataIssue('無法識別研究模型。')
    data=frame if threshold is None else frame.loc[frame['target_pts']>=threshold]
    columns=['model_variant','n','mae','rmse','mean_residual']
    if data.empty:return pd.DataFrame(columns=columns)
    y=data['target_pts'].to_numpy(dtype=float);rows=[]
    for m in models:
        p=data['pred_'+m].to_numpy(dtype=float)
        if not np.isfinite(y).all() or not np.isfinite(p).all():raise DataIssue('比較分數包含無效數值。')
        e=y-p;rows.append(dict(model_variant=m,n=len(e),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),mean_residual=float(e.mean())))
    return pd.DataFrame(rows,columns=columns)


def revision(root):
    folder=Path(root)/'outputs/dashboard_comparison'
    try:return tuple((p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(folder.iterdir()) if p.is_file())
    except OSError as exc:raise DataIssue('研究比較資料尚未備齊。') from exc


def load_research(root):
    folder=Path(root)/'outputs/dashboard_comparison'
    try:
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest['schema']!=1 or manifest['models']!=MODELS or set(manifest['datasets'])!=set(EXPECTED):raise DataIssue('研究比較版本不相符。')
        frames={}
        for season,periods in EXPECTED.items():
            if set(manifest['datasets'][season])!=set(periods):raise DataIssue('研究比較期間不相符。')
            frames[season]={}
            for period,(start,end,n) in periods.items():
                info=manifest['datasets'][season][period];filename=info['file']
                if Path(filename).name!=filename or not filename.endswith('.csv'):raise DataIssue('無效的比較資料路徑。')
                path=folder/filename
                if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:raise DataIssue('研究比較資料已變動，請確認同一版本的完整成果。')
                frame=pd.read_csv(path)
                validate_predictions(frame,expected_n=n,start=start,end=end,models=MODELS)
                if not np.isfinite(frame['actual_minutes'].to_numpy(dtype=float)).all() or (frame['actual_minutes']<=0).any():raise DataIssue('研究比賽的上場分鐘無法核對。')
                for group,threshold in [('all',None),('20+',20),('30+',30)]:
                    stats=research_metrics(frame,['baseline']+MODELS,threshold)
                    for r in stats.to_dict('records'):
                        saved=info['metrics'][group][r['model_variant']]
                        if saved['n']!=r['n'] or not np.allclose([saved[k] for k in ['mae','rmse','mean_residual']],[r[k] for k in ['mae','rmse','mean_residual']],rtol=1e-12,atol=1e-12):raise DataIssue('研究比較指標與逐場資料不一致。')
                frames[season][period]=frame.sort_values(['GAMEDATE','player_id','game_id']).reset_index(drop=True)
        return ResearchBundle(frames,manifest)
    except DataIssue:raise
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:raise DataIssue('無法讀取完整研究比較資料，請確認部署檔案齊全。') from exc
