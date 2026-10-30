"""Original nine-feature model only; no prior or target fields at inference."""
import numpy as np
from xgboost import XGBRegressor
from scripts.experiment_recent_changes import BASE
from scripts.recent_game_features import canonical_date


def base_matrix(rows):
    if not rows:raise ValueError('Empty matrix')
    x=np.asarray([[float(r[c]) for c in BASE] for r in rows])
    if not np.isfinite(x).all():raise ValueError('Nonfinite base feature')
    return x


def fit_base(rows,params,budget):
    x=base_matrix(rows);y=np.asarray([float(r['target_pts']) for r in rows]);keys=set()
    for r in rows:
        day=canonical_date(r['GAMEDATE']).isoformat();key=r['player_id'],r['game_id']
        if not '2024-10-22'<=day<='2025-01-31' or r['split']!='train' or key in keys:raise ValueError('Wrong training date, split or duplicate')
        keys.add(key)
    if not np.isfinite(y).all() or (y<0).any():raise ValueError('Invalid training labels')
    model=XGBRegressor(**params);budget.consume();model.fit(x,y)
    return model
