"""Research-only product model: prior-date features -> minutes and points/minute."""
import numpy as np
from xgboost import XGBRegressor

from scripts.experiment_recent_changes import BASE,matrix
from scripts.recent_change_features import NEW_FEATURES
from scripts.recent_game_features import canonical_date,number

FEATURES=BASE+NEW_FEATURES


def attach_minute_targets(rows,history):
    source={}; result=[]; seen=set()
    for h in history:
        # Refuse to index consumed test outcomes, even if the caller supplies them.
        if canonical_date(h['GAMEDATE']).isoformat()>'2026-02-28':
            continue
        key=h['player_id'],h['game_id']
        if key in source:
            raise ValueError('Duplicate minute source')
        source[key]=h
    for row in rows:
        day=canonical_date(row['GAMEDATE']).isoformat(); split=row['split']
        if not ((split=='train' and '2025-10-21'<=day<='2026-01-31') or
                (split=='validation' and '2026-02-01'<=day<='2026-02-28')):
            raise ValueError('Only original development targets are allowed')
        key=row['player_id'],row['game_id']
        if key in seen or key not in source:
            raise ValueError('Duplicate sample or missing minute source')
        seen.add(key); h=source[key]
        minute=number(h['MIN']); point=number(h['PTS'],integer=True)
        if h['GAMEDATE']!=day or row['game_id'].rsplit('|',1)[-1]!=day or minute in (None,0) or point is None or point!=float(row['target_pts']):
            raise ValueError('Invalid minute target or source label mismatch')
        result.append(dict(row,target_minutes=minute,target_ppm=point/minute))
    return result


class TwoStageModel:
    """Both predictors share fixed causal features; no label or fitted stacking feature."""
    def __init__(self,params):
        self.params=dict(params)
        self.columns=list(FEATURES)
        self.research_only=True

    def fit(self,rows,budget):
        x=matrix(rows,self.columns)
        minutes=np.asarray([float(r['target_minutes']) for r in rows])
        points=np.asarray([float(r['target_pts']) for r in rows])
        days=[canonical_date(r['GAMEDATE']).isoformat() for r in rows]
        if (not np.isfinite(minutes).all() or not np.isfinite(points).all() or
                (minutes<=0).any() or (points<0).any() or
                min(days)<'2025-10-21' or max(days)>'2026-01-31' or any(r['split']!='train' for r in rows)):
            raise ValueError('Invalid or out-of-training labels')
        rate=points/minutes
        weights=minutes/minutes.mean()
        if not np.isfinite(rate).all() or not np.isfinite(weights).all():
            raise ValueError('Nonfinite rate labels/weights')
        self.minutes_model=XGBRegressor(**self.params); self.rate_model=XGBRegressor(**self.params)
        budget.consume(); self.minutes_model.fit(x,minutes)
        budget.consume(); self.rate_model.fit(x,rate,sample_weight=weights)
        self.trained_through=max(days)
        self.training_n=len(rows)
        self.weight_normalizer=float(minutes.mean())
        return self

    def predict_components(self,rows):
        # Deliberately selects the frozen feature whitelist, never target_minutes/target_pts.
        x=matrix(rows,self.columns)
        raw_minutes=np.asarray(self.minutes_model.predict(x),dtype=float)
        raw_rate=np.asarray(self.rate_model.predict(x),dtype=float)
        if raw_minutes.shape!=(len(rows),) or raw_rate.shape!=(len(rows),) or not np.isfinite(np.concatenate([raw_minutes,raw_rate])).all():
            raise ValueError('Invalid model components')
        minutes=np.maximum(raw_minutes,0); rate=np.maximum(raw_rate,0)
        points=minutes*rate
        if not np.isfinite(points).all():
            raise ValueError('Nonfinite product prediction')
        return dict(minutes=minutes,rate=rate,points=points,raw_minutes=raw_minutes,raw_rate=raw_rate)
