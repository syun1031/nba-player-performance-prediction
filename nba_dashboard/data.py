"""Verified, read-only display data. No model inference or fitting."""
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS=[name+'_'+group for name in ['linear','tree','xgboost'] for group in ['basic','starter','teammates']]
PERIODS={'test':('2026-03-01','2026-04-12',6979),'validation':('2026-02-01','2026-02-28',3462)}
FILES={
    'candidate':'outputs/week2/final_test/evaluated_candidate.json',
    'protocol':'outputs/week2/final_test/test_protocol.json',
    'test_evidence':'outputs/week2/final_test/validation.json',
    'test_predictions':'outputs/week2/final_test/predictions_test.csv',
    'test_metrics':'outputs/week2/final_test/model_metrics.csv',
    'validation_evidence':'outputs/week2/time_validation_tuning/validation.json',
    'validation_predictions':'outputs/week2/time_validation_tuning/predictions_february.csv',
    'validation_metrics':'outputs/week2/time_validation_tuning/february_metrics.csv',
    'history':'outputs/nba_2025_26/nba_2025_26_regular_season_player_game_logs.csv',
    'history_evidence':'outputs/week1/features_baseline/validation.json',
}


class DataIssue(ValueError):
    """An actionable data issue that is safe to show without a traceback."""


@dataclass
class DashboardBundle:
    primary: str
    predictions: dict
    history: pd.DataFrame
    version: str
    candidate: dict


def file_revision(root):
    try:
        return tuple((name,(Path(root)/path).stat().st_mtime_ns,(Path(root)/path).stat().st_size) for name,path in FILES.items())
    except OSError as exc:
        raise DataIssue('回測資料尚未備齊，請確認資料檔完整後再開啟。') from exc


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_hash(mapping,relative_path):
    normalized={}
    for path,value in mapping.items():
        key=path.replace('\\','/')
        if key in normalized and normalized[key]!=value:
            raise DataIssue('資料的版本紀錄互相衝突，暫時無法展示。')
        normalized[key]=value
    return normalized[str(relative_path).replace('\\','/')]


def validate_predictions(frame,*,expected_n,start,end,models):
    required=['player_id','game_id','GAMEDATE','target_pts','pred_baseline']+['pred_'+m for m in models]
    if not frame.columns.is_unique or not set(required)<=set(frame.columns) or len(frame)!=expected_n:
        raise DataIssue('回測資料的欄位或樣本數不一致，請使用同一版本的完整成果。')
    if frame[required].isna().any().any() or frame.duplicated(['player_id','game_id']).any():
        raise DataIssue('回測資料有缺漏或重複紀錄，暫時無法展示。')
    days=frame['GAMEDATE'].astype(str)
    if (not days.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all()
        or pd.to_datetime(days,format='%Y-%m-%d',errors='coerce').isna().any() or not days.between(start,end).all()
        or not frame['game_id'].str.rsplit('|',n=1).str[-1].eq(days).all()):
        raise DataIssue('回測日期不屬於選定的資料期間。')
    numeric=frame[['target_pts','pred_baseline']+['pred_'+m for m in models]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (frame['target_pts']<0).any():
        raise DataIssue('回測分數包含缺失或無效數值，暫時無法展示。')


def recent_history(history,player,target_day,limit=10):
    days=pd.to_datetime(history['GAMEDATE'],format='%Y-%m-%d')
    minutes=pd.to_numeric(history['MIN'])
    result=history.loc[(history['PLAYER']==player)&(days<pd.Timestamp(target_day))&(minutes>0)].copy()
    result['GAMEDATE']=pd.to_datetime(result['GAMEDATE'],format='%Y-%m-%d')
    return result.sort_values('GAMEDATE',kind='stable').tail(limit).reset_index(drop=True)


def comparison_metrics(frame,models,threshold=None):
    data=frame if threshold is None else frame.loc[frame['target_pts']>=threshold]
    fields=['model_variant','n','mae','rmse','mean_residual']
    if data.empty or not models:
        return pd.DataFrame(columns=fields)
    actual=data['target_pts'].to_numpy(dtype=float)
    rows=[]
    for model in models:
        if model not in ['baseline']+MODELS:
            raise DataIssue('無法識別這個比較版本。')
        predicted=data['pred_'+model].to_numpy(dtype=float)
        if not np.isfinite(predicted).all():
            raise DataIssue('比較資料有無效預測，暫時無法展示。')
        errors=actual-predicted
        rows.append(dict(model_variant=model,n=len(data),mae=float(np.abs(errors).mean()),
                         rmse=float(np.sqrt(np.mean(errors**2))),mean_residual=float(errors.mean())))
    return pd.DataFrame(rows,columns=fields)


def prediction_table(frame,primary):
    data=frame.copy()
    residual=data['target_pts']-data['pred_'+primary]
    return pd.DataFrame({
        '比賽日期':data['GAMEDATE'],'對戰':data['game_id'].str.rsplit('|',n=1).str[0],
        '實際得分':data['target_pts'],'模型預測':data['pred_'+primary],'五場均值':data['pred_baseline'],
        '絕對誤差':residual.abs(),
        '預測狀況':['低估 %.2f 分'%r if r>0 else '高估 %.2f 分'%(-r) if r<0 else '與實際相同' for r in residual],
    }).reset_index(drop=True)


def load_bundle(root):
    root=Path(root)
    try:
        file_revision(root)
        paths={k:root/v for k,v in FILES.items()}
        read_json=lambda key:json.loads(paths[key].read_text(encoding='utf-8'))
        test_evidence=read_json('test_evidence'); val_evidence=read_json('validation_evidence')
        if not test_evidence['verified'] or not test_evidence['test_consumed'] or not val_evidence['verified']:
            raise DataIssue('目前沒有通過核對的完整回測成果。')
        for key in ['candidate','protocol','test_predictions','test_metrics']:
            relative=str(paths[key].relative_to(root/'outputs/week2/final_test'))
            if _sha(paths[key])!=expected_hash(test_evidence['output_sha256'],relative):
                raise DataIssue('回測成果已變動，請先確認資料版本一致後再展示。')
        for key in ['validation_evidence','validation_predictions','validation_metrics']:
            if _sha(paths[key])!=expected_hash(test_evidence['source_sha256'],paths[key].relative_to(root)):
                raise DataIssue('二月驗證資料與已鎖定模型的版本不同，暫時無法展示。')
        for key in ['validation_predictions','validation_metrics']:
            relative=str(paths[key].relative_to(root/'outputs/week2/time_validation_tuning'))
            if _sha(paths[key])!=expected_hash(val_evidence['output_sha256'],relative):
                raise DataIssue('驗證成果已變動，請先確認資料版本一致後再展示。')
        # The raw historical snapshot must match the one verified by the original feature build.
        raw_evidence=read_json('history_evidence')
        if _sha(paths['history'])!=expected_hash(raw_evidence['source_sha256'],paths['history'].relative_to(root)):
            raise DataIssue('近期歷史資料與回測版本不一致，暫時無法展示。')
        candidate=read_json('candidate'); protocol=read_json('protocol'); primary=protocol['primary_model']
        if primary!=test_evidence['primary_model'] or primary!=candidate['result']['model']+'_'+candidate['result']['features']:
            raise DataIssue('已鎖定模型的紀錄不一致，暫時無法展示。')
        predictions={}
        for period,(start,end,n) in PERIODS.items():
            frame=pd.read_csv(paths[period+'_predictions'],encoding='utf-8-sig')
            validate_predictions(frame,expected_n=n,start=start,end=end,models=MODELS)
            predictions[period]=frame.sort_values(['GAMEDATE','player_id','game_id']).reset_index(drop=True)
        saved=pd.read_csv(paths['test_metrics'])
        measured=comparison_metrics(predictions['test'],['baseline']+MODELS)
        for r in measured.to_dict('records'):
            reference=saved.loc[saved['model_variant']==r['model_variant']]
            if len(reference)!=1 or int(reference.iloc[0]['n'])!=r['n'] or not np.allclose(
                    [r['mae'],r['rmse']],[reference.iloc[0]['mae'],reference.iloc[0]['rmse']],rtol=1e-12,atol=1e-12):
                raise DataIssue('測試指標與逐筆紀錄不一致，暫時無法展示。')
        saved_val=pd.read_csv(paths['validation_metrics'])
        saved_val=saved_val.loc[saved_val['split']=='validation'].copy()
        saved_val['model_variant']=saved_val.apply(lambda r:'baseline' if r['model']=='baseline' else r['model']+'_'+r['features'],axis=1)
        for r in comparison_metrics(predictions['validation'],['baseline']+MODELS).to_dict('records'):
            reference=saved_val.loc[saved_val['model_variant']==r['model_variant']]
            if len(reference)!=1 or int(reference.iloc[0]['n'])!=r['n'] or not np.allclose(
                    [r['mae'],r['rmse']],[reference.iloc[0]['mae'],reference.iloc[0]['rmse']],rtol=1e-12,atol=1e-12):
                raise DataIssue('驗證指標與逐筆紀錄不一致，暫時無法展示。')
        history=pd.read_csv(paths['history'],usecols=['PLAYER','GAMEDATE','MIN','PTS'])
        if history[['PLAYER','GAMEDATE','MIN','PTS']].isna().any().any() or not np.isfinite(history[['MIN','PTS']].to_numpy(dtype=float)).all():
            raise DataIssue('近期歷史資料不完整，暫時無法展示。')
        version=datetime.fromisoformat(protocol['created_at_utc']).date().isoformat()
        return DashboardBundle(primary,predictions,history,version,candidate)
    except DataIssue:
        raise
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        raise DataIssue('無法讀取完整回測資料，請確認資料版本一致後再試。') from exc
