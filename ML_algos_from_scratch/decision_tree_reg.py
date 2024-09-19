import pandas as pd
import numpy as np


class Node:
    """
    Represents a node in the decision tree.

    Attributes:
        feature (str): The feature used for splitting at this node.
        threshold (float): The threshold value for the split.
        left (Node): The left child node.
        right (Node): The right child node.
        side (str): Indicates whether this node is a 'left' or 'right' child.
        value (float): The prediction value if this is a leaf node.
    """
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
    """
    A custom implementation of a Decision Tree Regressor.

    This class provides functionality to build and use a decision tree for regression tasks.
    It supports various hyperparameters for tree construction and uses mean squared error
    as the splitting criterion.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        max_leafs (int): Maximum number of leaf nodes.
        bins (int): Number of bins for continuous features (if specified).
        leafs_cnt (int): Current count of leaf nodes.
        tree (Node): The root node of the decision tree.
        col_thresholds (dict): Stores potential split thresholds for each feature.
        fi (dict): Feature importances.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
        bins: int = None
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
        """
        Returns a string representation of the MyTreeReg object.

        Returns:
            str: A string containing the class name and its parameters.
        """
        atts = ', '.join(f'{k}={v}' for k, v in vars(self).items())
        return f'MyTreeReg class: {atts}'

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        """
        Finds the best split for a node.

        Args:
            X (pd.DataFrame): The feature dataset.
            y (pd.Series): The target values.

        Returns:
            tuple: The best feature, split value, and information gain.
        """
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
                        )[1][1:-1]

            splits = []
            for t in self.col_thresholds[col]:
                y_left = y[X[col] <= t]
                y_right = y[X[col] > t]

                if len(y_left) > 0 and len(y_right) > 0:
                    gain = (
                        mse_0 - len(y_left)/len(y) * np.var(y_left)
                        - len(y_right)/len(y) * np.var(y_right)
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

    def fit(self, X: pd.DataFrame, y: pd.Series, init_data_len: int = None):
        """
        Builds the decision tree from the training set (X, y).

        Args:
            X (pd.DataFrame): The input samples.
            y (pd.Series): The target values.
            init_data_len (int, optional): Initial data length for feature importance calculation.
        """
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
                or len(y_node) < self.min_samples_split
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

            if len(X_left) == 0 or len(X_right) == 0:
                node.value = y_node.mean()
                node.side = side
                return node

            if init_data_len is None:
                self.fi[col_name] += len(y_node)/len(y) * gain
            else:
                self.fi[col_name] += len(y_node)/init_data_len * gain

            node.feature = col_name
            node.threshold = split_value
            self.leafs_cnt += 1

            node.left = _grow_tree(
                node.left, X_left, y_left, 'left', depth=depth+1
                )
            node.right = _grow_tree(
                node.right, X_right, y_right, 'right', depth=depth+1
            )

            return node

        self.tree = _grow_tree(self.tree, X, y)

    def _predict_row(self, row):
        """
        Predicts the target value for a single input row.

        Args:
            row (pd.Series): A single input sample.

        Returns:
            float: The predicted target value.
        """
        node = self.tree
        while node.feature is not None:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X: pd.DataFrame):
        """
        Predicts target values for samples in X.

        Args:
            X (pd.DataFrame): The input samples.

        Returns:
            pd.Series: The predicted target values.
        """
        _X = X.copy()
        return _X.apply(self._predict_row, axis=1)

    def print_tree(self, node=None, depth=0):
        """
        Prints the decision tree structure.

        Args:
            node (Node, optional): The current node to print. Defaults to the root node.
            depth (int, optional): The current depth in the tree. Defaults to 0.
        """
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

# Description of the classes and their functionality:

# Node class:
# This class represents a node in our decision tree. Each node can be either a decision node
# (with a feature and threshold for splitting) or a leaf node (with a prediction value).
# It's the fundamental building block of our tree structure.

# MyTreeReg class:
# This is our main Decision Tree Regressor class. It's a custom implementation that allows
# you to create, train, and use a decision tree for regression tasks. Here's what it can do:

# 1. Build a tree: Use the 'fit' method to construct a decision tree from your training data.
# 2. Make predictions: Once trained, you can use 'predict' to estimate target values for new data.
# 3. Visualize the tree: The 'print_tree' method lets you see the structure of your trained tree.
# 4. Customize your tree: You can adjust various parameters like max depth, minimum samples to split,
#    maximum number of leaves, and even bin continuous features if needed.

# This implementation is great for learning about decision trees or for when you need a customizable,
# transparent regressor. It uses mean squared error as the splitting criterion to grow the tree.
# The feature importance is also calculated during the tree construction.

# Enjoy exploring it if you're a beginner.
# Otherwise, I'm flattered you've made it this far!
