"""Train-only tail weights and unweighted diagnostics with fixed high-score alerts."""
from collections import defaultdict
import numpy as np
from sklearn.metrics import average_precision_score

from scripts.experiment_recent_changes import BASE,matrix
from scripts.recent_change_features import NEW_FEATURES
from scripts.opponent_context import CONTEXT_FEATURES
from scripts.same_conference_h2h import H2H_FEATURES,h2h_matrix
from scripts.compare_models import metrics
from scripts.time_validation_tuning import stratified_interval

COMBINED=BASE+NEW_FEATURES+CONTEXT_FEATURES+H2H_FEATURES
VARIANTS={
    'basic_w2':dict(base='basic',multiplier=2,columns=list(BASE)),
    'basic_w3':dict(base='basic',multiplier=3,columns=list(BASE)),
    'combined_w2':dict(base='combined',multiplier=2,columns=list(COMBINED)),
    'combined_w3':dict(base='combined',multiplier=3,columns=list(COMBINED)),
}
PAIRS=[('basic_unweighted','basic_w2'),('basic_unweighted','basic_w3'),
       ('combined_unweighted','combined_w2'),('combined_unweighted','combined_w3'),
       ('basic_w2','combined_w2'),('basic_w3','combined_w3')]


def high_score_weights(actual,multiplier):
    y=np.asarray(actual,dtype=float)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all() or (y<0).any() or not np.isfinite(multiplier) or multiplier<1:
        raise ValueError('Invalid training labels or high-score multiplier')
    raw=np.where(y>=30,float(multiplier),1.)
    normalizer=float(raw.mean())
    return raw/normalizer,normalizer


def input_matrix(rows,variant):
    spec=VARIANTS[variant]
    return matrix(rows,spec['columns']) if spec['base']=='basic' else h2h_matrix(rows,spec['columns'])


def aligned(actual,predicted):
    y,p=[np.asarray(x,dtype=float) for x in (actual,predicted)]
    if y.ndim!=1 or y.shape!=p.shape or not len(y) or not np.isfinite(np.concatenate([y,p])).all() or (y<0).any():
        raise ValueError('Invalid or unaligned scoring inputs')
    return y,p


def alert_metrics(actual,predicted):
    y,p=aligned(actual,predicted); truth=y>=30; alert=p>=30
    tp=int((truth&alert).sum()); fp=int((~truth&alert).sum())
    fn=int((truth&~alert).sum()); tn=int((~truth&~alert).sum())
    return dict(n=len(y),actual_high_n=tp+fn,predicted_high_n=tp+fp,tp=tp,fp=fp,fn=fn,tn=tn,
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        false_positive_rate=fp/(fp+tn) if fp+tn else None,prevalence=float(truth.mean()),
        average_precision=float(average_precision_score(truth,p)) if truth.any() else None)


def point_group_scores(actual,predicted):
    y,p=aligned(actual,predicted)
    groups={'all':np.ones(len(y),bool),'under30':y<30,'20+':y>=20,'30+':y>=30,
            '0_to_under10':y<10,'10_to_under20':(y>=10)&(y<20),'20_to_under30':(y>=20)&(y<30)}
    result=[]
    for name,mask in groups.items():
        score=metrics(y[mask],p[mask]) if mask.any() else dict(n=0,mae=None,rmse=None)
        residual=y[mask]-p[mask]
        result.append(dict(group=name,**score,mean_residual=float(residual.mean()) if mask.any() else None,
            overprediction_fraction=float((residual<0).mean()) if mask.any() else None))
    return result


def paired_comparison(rows,first,second,group):
    if group not in ['all','30+']: raise ValueError('Unknown fixed comparison group')
    selected=rows if group=='all' else [r for r in rows if float(r['target_pts'])>=30]
    dates=defaultdict(set)
    for r in selected: dates[r['period']].add(r['GAMEDATE'])
    strata=set(r['period'] for r in rows)
    y=[float(r['target_pts']) for r in selected]; old=[float(r['pred_'+first]) for r in selected]; new=[float(r['pred_'+second]) for r in selected]
    if not y or any(len(dates[s])<2 for s in strata):
        delta=float((np.abs(np.asarray(new)-y)-np.abs(np.asarray(old)-y)).mean()) if y else None
        return dict(status='insufficient_dates',n=len(y),delta_mae=delta,delta_rmse=None,
                    lower_95=None,upper_95=None,iterations=0,seed=42)
    return dict(status='estimated',n=len(y),**stratified_interval(y,old,new,
        [r['GAMEDATE'] for r in selected],[r['period'] for r in selected]))
