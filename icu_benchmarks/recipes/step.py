from copy import deepcopy
from sklearn.preprocessing import StandardScaler

from icu_benchmarks.recipes.selector import all_predictors


class Step():
    def __init__(self, sel=all_predictors()) -> None:
        self.sel = sel
        self.columns = []
        self._trained = False
        self._group = True

    @property
    def trained(self):
        return self._trained

    @property
    def group(self):
        return self._group

    def fit(self, data):
        pass

    def transform(self, data):
        pass

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def __repr__(self):
        repr = self.desc + ' for '

        if not self.trained:
            repr += str(self.sel)
        else:
            repr += str(self.columns) if len(self.columns) < 3 else str(self.columns[:2] + ['...']) # FIXME: remove brackets
            repr += ' [trained]'

        return repr


class StepImputeFill(Step):
    def __init__(self, sel=all_predictors(), value=None, method=None, limit=None):
        super().__init__(sel)
        self.desc = f'Impute with {method if method else value}'
        self.value = value
        self.method = method
        self.limit = limit

    def fit(self, data):
        self.columns = self.sel(data.obj)
        self._trained = True

    def transform(self, data):
        new_data = data.obj # FIXME: also deal with ungrouped DataFrames
        new_data[self.columns] = \
            data[self.columns].fillna(self.value, method=self.method, axis=0, limit=self.limit)
        return new_data


class StepScale(Step):
    def __init__(self, sel=all_predictors(), with_mean=True, with_std=True):
        super().__init__(sel)
        self.desc = f'Scale with mean ({with_mean}) and std ({with_std})'
        self.with_mean = with_mean
        self.with_std = with_std
        self._group = False

    def fit(self, data):
        self.columns = self.sel(data)
        self.scalers = {
            c: StandardScaler(copy=True, with_mean=self.with_mean, with_std=self.with_std).fit(data[c].values[:, None])
            for c in self.columns
        }
        self._trained = True

    def transform(self, data):
        new_data = data
        for c, sclr in self.scalers.items():
            new_data[c] = sclr.transform(data[c].values[:, None])
        return new_data


class StepHistorical(Step):
    def __init__(self, sel=all_predictors(), fun='max', suffix=None, role='predictor'):
        super().__init__(sel)
        self.desc = f'Create historical {fun}'
        self.fun = fun
        if suffix is None:
            suffix = fun
        self.suffix = suffix
        self.role = role

    def fit(self, data):
        self.columns = self.sel(data.obj)
        self._trained = True

    def transform(self, data):
        new_data = data.obj # FIXME: also deal with ungrouped DataFrames
        new_columns = [c + '_' + self.suffix for c in self.columns]

        if self.fun == 'max':
            res = data[self.columns].cummax(skipna=True)
        elif self.fun == 'min':
            res = data[self.columns].cummin(skipna=True)
        new_data[new_columns] = res

        for nc in new_columns:
            new_data.add_role(nc, self.role)

        return new_data


class StepSklearn(Step):
    # TODO docstring
    def __init__(self, sel=all_predictors(), sklearn_transform=None, columnwise=False, is_in_place=True):
        super().__init__(sel)
        self.desc = f'Use sklearn transform {sklearn_transform.__class__.__name__}'
        self.sklearn_transform = sklearn_transform
        self.columnwise = columnwise
        self.is_in_place = is_in_place
        self._group = False

    def fit(self, data):
        self.columns = self.sel(data)
        if self.columnwise:
            self._transformers = {}
            for col in self.columns:
                # copy the transformer so we keep the distinct fit for each column and don't just refit
                self._transformers[col] = deepcopy(self.sklearn_transform.fit(data[col]))
        else:
            self.sklearn_transform.fit(data[self.columns])
        self._trained = True

    def transform(self, data):
        new_data = data

        if self.columnwise:
            for col in self.columns:
                new_cols = self._transformers[col].transform(new_data[col])
                col_names = col if self.is_in_place else [f'{self.sklearn_transform.__class__.__name__}_{col}_{i+1}' for i in range(new_cols.shape[1])]
                new_data[col_names] = new_cols
        else:
            new_cols = self.sklearn_transform.transform(new_data[self.columns])
            col_names = self.columns if self.is_in_place else [f'{self.sklearn_transform.__class__.__name__}_{i+1}' for i in range(new_cols.shape[1])]
            new_data[col_names] = new_cols
        
        return new_data
