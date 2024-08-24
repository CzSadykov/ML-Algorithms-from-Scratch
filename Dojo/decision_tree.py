import pandas as pd
import numpy as np

class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        side=None,
        value=None
        ):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.side = side
        self.right = right
        self.value = value


class MyTreeReg:
    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        max_leafs=20
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        
        
    def __repr__(self):
        atts = ', '.join(f'{k}={v}' for k, v in vars(self).items())
        return f'MyTreeReg class: {atts}'
    
    
    def get_best_split(X, y):
        col_splits = {}
        mse_0 = np.var(y)

        for col in X.columns:
            vals = np.sort(X[col].unique())
            splits = []
            thresholds = (vals[1:] + vals[:-1]) / 2

            for t in thresholds:
                X_left = X[X[col] <= t]
                y_left = y[X[col] <= t]
                X_right = X[X[col] > t]
                y_right = y[X[col] > t]     

                gain = mse_0 - y_left.size/y.size * np.var(y_left) - y_right.size/y.size * np.var(y_right)
                splits.append((t, gain))

            col_splits[col] = max(splits, key=lambda x: x[1])

        col_name, (split_value, gain) = max(col_splits.items(), key=lambda x: x[1][1])

        return col_name, split_value, gain


    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X: pd.DataFrame, y: pd.Series):
        if len(y) <= self.min_samples_split or len(X.columns) <= 1 or self.max_depth <= 0:
            return Node(value=y.mean(), ) 

    
        


