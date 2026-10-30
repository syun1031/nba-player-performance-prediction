"""Learn points minus causal last-five mean; add it back only at prediction."""
import numpy as np
from xgboost import XGBRegressor

from scripts.high_score_weighting import COMBINED,high_score_weights
from scripts.same_conference_h2h import h2h_matrix
from scripts.recent_game_features import canonical_date

COLUMNS=list(COMBINED)


class ResidualOffsetModel:
    def __init__(self,params,multiplier=1):
        if multiplier not in (1,2): raise ValueError('Only the two preregistered multipliers are allowed')
        self.params=dict(params); self.multiplier=multiplier; self.columns=list(COLUMNS); self.research_only=True

    def fit(self,rows,budget):
        x=h2h_matrix(rows,self.columns)
        labels=np.asarray([float(r['target_pts']) for r in rows])
        baseline=np.asarray([float(r['pts_mean_5']) for r in rows])
        days=[canonical_date(r['GAMEDATE']).isoformat() for r in rows]
        keys=[(r['player_id'],r['game_id']) for r in rows]
        if (len(set(keys))!=len(rows) or not np.isfinite(baseline).all() or (baseline<0).any()
            or min(days)<'2025-10-21' or max(days)>'2026-01-31' or any(r['split']!='train' for r in rows)):
            raise ValueError('Invalid baseline, cohort or training date')
        weights,self.weight_normalizer=high_score_weights(labels,self.multiplier)
        delta=labels-baseline
        self.model=XGBRegressor(**self.params); budget.consume()
        self.model.fit(x,delta,sample_weight=weights)
        self.trained_through=max(days); self.training_n=len(rows)
        self.train_high_n=int((labels>=30).sum()); self.target_name='target_pts_minus_pts_mean_5'
        return self

    def predict_components(self,rows):
        x=h2h_matrix(rows,self.columns)
        baseline=np.asarray([float(r['pts_mean_5']) for r in rows])
        delta=np.asarray(self.model.predict(x),dtype=float)
        if delta.shape!=(len(rows),) or not np.isfinite(delta).all() or not np.isfinite(baseline).all() or (baseline<0).any():
            raise ValueError('Invalid baseline or predicted delta')
        raw_points=baseline+delta
        if not np.isfinite(raw_points).all(): raise ValueError('Nonfinite summed prediction')
        return dict(baseline=baseline,delta=delta,raw_points=raw_points,points=np.maximum(raw_points,0))
