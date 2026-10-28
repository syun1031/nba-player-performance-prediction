"""Deterministic multi-hot teammate encoding with train-only vocabulary."""


def validate_names(names):
    if not isinstance(names,(list,tuple)):
        raise ValueError('Teammates must be a list of names')
    if not all(isinstance(n,str) and n and n==n.strip() for n in names):
        raise ValueError('Teammate names must be nonempty exact strings')
    if len(set(names))!=len(names):
        raise ValueError('Duplicate teammate name')
    return list(names)


class TeammateEncoder:
    def __init__(self):
        self.classes_=None
        self._index=None

    def fit(self,X,y=None):
        classes=set();count=0
        for names in X:
            classes.update(validate_names(names));count+=1
        if count==0:
            raise ValueError('Cannot fit an empty training set')
        self.classes_=sorted(classes)
        self._index={name:i for i,name in enumerate(self.classes_)}
        return self

    def _require_fit(self):
        if self.classes_ is None:
            raise ValueError('Fit the encoder on training data first')

    def transform(self,X):
        self._require_fit()
        output=[]
        for names in X:
            values=[0]*(len(self.classes_)+1)
            for name in validate_names(names):
                index=self._index.get(name)
                if index is None:
                    values[-1]+=1
                else:
                    values[index]=1
            output.append(values)
        return output

    def fit_transform(self,X,y=None):
        rows=list(X)
        return self.fit(rows,y).transform(rows)

    def get_feature_names_out(self,input_features=None):
        self._require_fit()
        return [f'teammate_{i+1:04d}' for i in range(len(self.classes_))]+['teammate_unknown_count']

    def to_dict(self):
        self._require_fit()
        return dict(format_version=1,encoding='multi_hot',classes=list(self.classes_),
                    feature_names=self.get_feature_names_out(),unknown_policy='count_only_no_refit')

    @classmethod
    def from_dict(cls,state):
        if state.get('format_version')!=1 or state.get('encoding')!='multi_hot' or state.get('unknown_policy')!='count_only_no_refit':
            raise ValueError('Unsupported encoder format')
        names=validate_names(state['classes'])
        if names!=sorted(names):
            raise ValueError('Persisted vocabulary order is invalid')
        obj=cls()
        obj.classes_=names
        obj._index={name:i for i,name in enumerate(names)}
        if state.get('feature_names')!=obj.get_feature_names_out():
            raise ValueError('Persisted feature order does not match vocabulary')
        return obj

    def get_params(self,deep=True):
        return {}

    def set_params(self,**params):
        if params:
            raise ValueError(f'Unknown parameters: {sorted(params)}')
        return self
