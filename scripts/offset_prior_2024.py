from scripts.offset_prior_model import OffsetPriorModel,COLUMNS
from scripts.prior_season_profile import profile_matrix
from scripts.high_score_weighting import high_score_weights
from scripts.recent_game_features import canonical_date
from xgboost import XGBRegressor
import numpy as np

class OffsetPrior2024Model(OffsetPriorModel):
    def fit(self,rows,budget):
        x=profile_matrix(rows,self.columns)
        y=np.asarray([float(r['target_pts']) for r in rows]);base=np.asarray([float(r['pts_mean_5']) for r in rows])
        days=[canonical_date(r['GAMEDATE']).isoformat() for r in rows];keys=[(r['player_id'],r['game_id']) for r in rows]
        if not rows or len(keys)!=len(set(keys)) or len({(r['player_id'],r['GAMEDATE']) for r in rows})!=len(rows) or min(days)<'2024-10-22' or max(days)>'2025-01-31' or any(r['split']!='train' for r in rows):raise ValueError('Invalid 2024 training cohort')
        if not np.isfinite(base).all() or (base<0).any():raise ValueError('Invalid baseline')
        weights,self.weight_normalizer=high_score_weights(y,2)
        self.model=XGBRegressor(**self.params);budget.consume();self.model.fit(x,y-base,sample_weight=weights)
        self.training_n=len(rows);self.train_high_n=int((y>=30).sum());self.trained_through=max(days)
        self.target_name='target_pts_minus_pts_mean_5'
        return self
