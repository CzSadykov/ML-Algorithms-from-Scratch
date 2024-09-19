import pandas as pd
import numpy as np
import random
import copy
from ML_algos_from_scratch.lin_reg import MyLineReg
from ML_algos_from_scratch.knn_reg import MyKNNReg
from ML_algos_from_scratch.decision_tree_reg import MyTreeReg


class MyBaggingReg:
    def __init__(
            self,
            estimator=None,
            n_estimators: int = 10,
            max_samples: float = 1.0,
            random_state: int = 42
            ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators = []

    def fit(self, X: pd.DataFrame, y: pd.Series):

        random.seed(self.random_state)

        init_rows_cnt = range(X.shape[0])
        rows_sample_cnt = int(X.shape[0] * self.max_samples)

        sample_rows_dict = {}

        for i in range(self.n_estimators):
            sample_rows_dict[i] = random.choices(
                init_rows_cnt, k=rows_sample_cnt
                )

        for _ in range(self.n_estimators):
            sample_idx = sample_rows_dict[i]
            X_sample = X.iloc[sample_idx, :].reset_index(drop=True)
            y_sample = y.iloc[sample_idx].reset_index(drop=True)
            model = copy.copy(self.estimator)
            model.fit(X_sample, y_sample)
            self.estimators.append(model)

    def __repr__(self):
        atts = ', '.join([f'{k}={v}' for k, v in vars(self).items()])
        return f'MyBaggingReg class: {atts}'
