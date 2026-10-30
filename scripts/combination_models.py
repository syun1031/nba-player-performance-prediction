"""Four explicit ablation combinations; preserve the original two-stage class."""
import math

from scripts.two_stage_model import TwoStageModel
from scripts.experiment_recent_changes import BASE
from scripts.recent_change_features import NEW_FEATURES
from scripts.opponent_context import CONTEXT_FEATURES

CASES={
    'case12':dict(label='1＋2',columns=BASE+NEW_FEATURES,mode='two_stage',fits=2),
    'case13':dict(label='1＋3',columns=BASE+NEW_FEATURES+CONTEXT_FEATURES,mode='direct',fits=1),
    'case23':dict(label='2＋3',columns=BASE+CONTEXT_FEATURES,mode='two_stage',fits=2),
    'case123':dict(label='1＋2＋3',columns=BASE+NEW_FEATURES+CONTEXT_FEATURES,mode='two_stage',fits=2),
}


class CombinationTwoStage(TwoStageModel):
    def __init__(self,params,case):
        if case not in CASES or CASES[case]['mode']!='two_stage':
            raise ValueError('Case does not specify a two-stage model')
        super().__init__(params)
        self.case=case
        self.columns=list(CASES[case]['columns'])


def rank_cases(rows):
    keys=[r['variant'] for r in rows]
    if len(keys)!=4 or set(keys)!=set(CASES) or any(
            not math.isfinite(float(r[m])) for r in rows for m in ['mae','rmse']):
        raise ValueError('Ranking needs exactly four finite case results')
    return sorted(rows,key=lambda r:(float(r['mae']),float(r['rmse']),len(CASES[r['variant']]['columns']),
                                      CASES[r['variant']]['fits'],r['variant']))
