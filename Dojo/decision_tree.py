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
        max_leafs=20,
        bins=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1
        self.tree = None
        self.bins = bins
        self.col_thresholds = {}
        self.fi = {}

    def __repr__(self):
        atts = ', '.join(f'{k}={v}' for k, v in vars(self).items())
        return f'MyTreeReg class: {atts}'

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        mse_0 = np.var(y)
        col_split_gain = {}

        for col in X.columns:
            if col not in self.col_thresholds.keys():
                vals = np.sort(X[col].unique())
                if self.bins is None or len(vals) <= self.bins - 1:
                    self.col_thresholds[col] = (vals[1:] + vals[:-1]) / 2
                else:
                    self.col_thresholds[col] = np.histogram(
                        X[col], bins=self.bins
                        )[1]

            splits = []
            for t in self.col_thresholds[col]:
                y_left = y[X[col] <= t]
                y_right = y[X[col] > t]

                if len(y_left) > 0 and len(y_right) > 0:
                    gain = (
                        mse_0 - y_left.size/y.size * np.var(y_left)
                        - y_right.size/y.size * np.var(y_right)
                    )
                    splits.append((t, gain))

            if splits:
                col_split_gain[col] = max(splits, key=lambda x: x[1])
            else:
                col_split_gain[col] = (None, -float('inf'))

        if col_split_gain:
            col_name, (split_value, gain) = max(
                col_split_gain.items(), key=lambda x: x[1][1]
            )
            return col_name, split_value, gain
        else:
            return None, None, -float('inf')

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.tree = None
        self.fi = {col: 0 for col in X.columns}

        def _grow_tree(
            node,
            X_node: pd.DataFrame,
            y_node: pd.Series,
            side: str = 'root',
            depth: int = 0
        ):
            if node is None:
                node = Node()

            if (
                len(y_node.unique()) in [0, 1]
                or len(y_node) <= self.min_samples_split
                or (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs)
                or depth >= self.max_depth
            ):
                node.value = y_node.mean()
                node.side = side
                return node

            col_name, split_value, gain = self.get_best_split(X_node, y_node)

            X_left = X_node[X_node[col_name] <= split_value]
            y_left = y_node[X_node[col_name] <= split_value]
            X_right = X_node[X_node[col_name] > split_value]
            y_right = y_node[X_node[col_name] > split_value]

            if X_left.size == 0 or X_right.size == 0:
                node.value = y_node.mean()
                node.side = side
                return node

            self.fi[col_name] += y_node.size/y.size * gain

            node.feature = col_name
            node.threshold = split_value
            self.leafs_cnt += 1

            left = _grow_tree(
                node.left, X_left, y_left, 'left', depth=depth+1
                )
            right = _grow_tree(
                node.right, X_right, y_right, 'right', depth=depth+1
            )

            return Node(col_name, split_value, left, right)

        self.tree = _grow_tree(self.tree, X, y)

    def _predict_row(self, row):
        node = self.tree
        while node.feature is not None:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X: pd.DataFrame):
        _X = X.copy()
        return _X.apply(self._predict_row, axis=1)

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.tree
        if node.feature is not None:
            print(f"{' ' * depth}{node.feature} > {node.threshold}")
            if node.left is not None:
                self.print_tree(node.left, depth + 1)
            if node.right is not None:
                self.print_tree(node.right, depth + 1)
        else:
            print(f"{' ' * depth}{node.side} = {node.value}")
