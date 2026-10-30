"""Frozen-scope 25-feature residual model with original-score 2x tail weights."""
import numpy as np
from xgboost import XGBRegressor
from scripts.high_score_weighting import COMBINED,high_score_weights
from scripts.prior_season_profile import PROFILE_FEATURES,profile_matrix
from scripts.recent_game_features import canonical_date

COLUMNS=list(COMBINED)+list(PROFILE_FEATURES)


class OffsetPriorModel:
    def __init__(self,params):
        self.params=dict(params);self.columns=list(COLUMNS);self.multiplier=2;self.research_only=True

    def fit(self,rows,budget):
        x=profile_matrix(rows,self.columns);y=np.asarray([float(r['target_pts']) for r in rows]);base=np.asarray([float(r['pts_mean_5']) for r in rows])
        days=[canonical_date(r['GAMEDATE']).isoformat() for r in rows];keys=[(r['player_id'],r['game_id']) for r in rows]
        if len(keys)!=len(set(keys)) or len({(r['player_id'],r['GAMEDATE']) for r in rows})!=len(rows) or min(days)<'2025-10-21' or max(days)>'2026-01-31' or any(r['split']!='train' for r in rows):raise ValueError('Invalid training date, split or keys')
        if not np.isfinite(base).all() or (base<0).any():raise ValueError('Invalid five-game baseline')
        weights,self.weight_normalizer=high_score_weights(y,2)
        delta=y-base
        self.model=XGBRegressor(**self.params);budget.consume();self.model.fit(x,delta,sample_weight=weights)
        self.training_n=len(rows);self.train_high_n=int((y>=30).sum());self.trained_through=max(days)
        self.target_name='target_pts_minus_pts_mean_5'
        return self

    def predict_components(self,rows):
        x=profile_matrix(rows,self.columns);base=np.asarray([float(r['pts_mean_5']) for r in rows]);delta=np.asarray(self.model.predict(x),dtype=float)
        if delta.shape!=(len(rows),) or not np.isfinite(base).all() or (base<0).any() or not np.isfinite(delta).all():raise ValueError('Invalid baseline or predicted delta')
        raw=base+delta
        if not np.isfinite(raw).all():raise ValueError('Nonfinite summed points')
        return dict(baseline=base,delta=delta,raw_points=raw,points=np.maximum(raw,0))
