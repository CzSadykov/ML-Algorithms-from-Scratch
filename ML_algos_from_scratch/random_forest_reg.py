import pandas as pd
import numpy as np
import random
from ML_algos_from_scratch.decision_tree_reg import MyTreeReg


class MyForestReg:
    """
    A beginner-friendly implementation of Random Forest Regressor.

    This class provides functionality to create and use a Random Forest for regression tasks.
    It allows customization of both forest-level and tree-level parameters.

    Attributes:
        n_estimators (int): Number of trees in the forest.
        max_features (float): Fraction of features to consider for each tree.
        max_samples (float): Fraction of samples to use for each tree.
        random_state (int): Seed for random number generation.
        max_depth (int): Maximum depth of each tree.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        max_leafs (int): Maximum number of leaf nodes allowed in each tree.
        bins (int): Number of bins for discretizing continuous features.
        oob_score (str): Type of out-of-bag score to compute.
        forest (list): List of decision trees in the forest.
        fi (dict): Feature importance scores.
    """

    def __init__(
            self,
            n_estimators: int = 10,
            max_features: float = 0.5,
            max_samples: float = 0.5,
            random_state: int = 42,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20,
            bins: int = 16,
            oob_score: str = None
            ):
        """
        Initialize the Random Forest Regressor.

        Args:
            n_estimators (int): Number of trees in the forest. Defaults to 10.
            max_features (float): Fraction of features to consider for each tree. Defaults to 0.5.
            max_samples (float): Fraction of samples to use for each tree. Defaults to 0.5.
            random_state (int): Seed for random number generation. Defaults to 42.
            max_depth (int): Maximum depth of each tree. Defaults to 5.
            min_samples_split (int): Minimum number of samples required to split an internal node. Defaults to 2.
            max_leafs (int): Maximum number of leaf nodes allowed in each tree. Defaults to 20.
            bins (int): Number of bins for discretizing continuous features. Defaults to 16.
            oob_score (str): Type of out-of-bag score to compute. Defaults to None.
        """
        # Forest parameters
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        # Tree parameteres
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.random_state = random_state
        self.forest = []
        self.fi = {}

        if oob_score in ['mae', 'mse', 'rmse', 'mape', 'r2']:
            self.oob_score = oob_score
        else:
            self.oob_score = None

        self.oob_score_ = None

    def _metric(self, y_true, y_pred):
        """
        Calculate the specified out-of-bag score metric.

        Args:
            y_true (array-like): True target values.
            y_pred (array-like): Predicted target values.

        Returns:
            float: Calculated metric value.
        """
        if self.oob_score == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif self.oob_score == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.oob_score == 'rmse':
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif self.oob_score == 'mape':
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        elif self.oob_score == 'r2':
            return (
                1 - np.sum((y_true - y_pred) ** 2)
                / np.sum((y_true - np.mean(y_true)) ** 2)
                )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the Random Forest Regressor to the training data.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target values.

        Returns:
            self: The fitted regressor.
        """
        rows, cols = X.shape
        self.leafs_cnt = 0

        random.seed(self.random_state)
        feature_list = list(X.columns)
        self.fi = {col: 0 for col in feature_list}

        init_cols = int(np.round(cols * self.max_features))
        init_rows = int(np.round(rows * self.max_samples))
        if self.oob_score:
            self.oob_predictions_ = {}
        # Create n_estimators trees
        for _ in range(self.n_estimators):
            cols_sample = random.sample(
                feature_list, init_cols
                )
            rows_sample = random.sample(
                range(rows), init_rows
                )
            X_sample = X.iloc[rows_sample][cols_sample]
            y_sample = y.iloc[rows_sample]

            tree = MyTreeReg(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_leafs=self.max_leafs,
                bins=self.bins
                )
            tree.fit(X_sample, y_sample, len(y))
            self.forest.append(tree)

            for col in cols_sample:
                self.fi[col] += tree.fi[col]

            self.leafs_cnt += tree.leafs_cnt

            if self.oob_score:
                oob_X = X.iloc[~X.index.isin(rows_sample)][cols_sample]
                oob_pred = tree.predict(oob_X)
                for idx, pred in zip(oob_X.index, oob_pred):
                    if idx not in self.oob_predictions_:
                        self.oob_predictions_[idx] = []
                    self.oob_predictions_[idx].append(pred)

        if self.oob_score:
            oob_preds = list(
                map(lambda x: np.mean(self.oob_predictions_[x]),
                    self.oob_predictions_.keys())
                    )
            oob_y = y[list(self.oob_predictions_.keys())].values
            self.oob_score_ = self._metric(oob_y, oob_preds)

        return self

    def predict(self, X: pd.DataFrame):
        """
        Predict target values for X.

        Args:
            X (pd.DataFrame): The input samples.

        Returns:
            np.array: The predicted target values.
        """
        predictions = []
        for tree in self.forest:
            predictions.append(tree.predict(X))
        return np.mean(predictions, axis=0)

    def __repr__(self):
        """
        Return a string representation of the MyForestReg object.

        Returns:
            str: A string containing the class name and its parameters.
        """
        atts = ', '.join([f'{k}={v}' for k, v in vars(self).items()])
        return f'MyForestReg class: {atts}'

# Description of the class and its functionality:

# MyForestReg is a beginner-friendly implementation of a Random Forest Regressor. It's designed to be
# easy to use while still offering a good degree of customization. Here's what it can do:

# 1. Create a forest: You can specify the number of trees, how many features and samples to use
#    for each tree, and various other parameters to control tree growth.

# 2. Train the forest: Use the 'fit' method to build your forest from training data. It'll create
#    multiple decision trees, each trained on a random subset of your data and features.

# 3. Make predictions: Once trained, you can use 'predict' to estimate target values for new data.

# 4. Evaluate performance: If you choose, it can calculate out-of-bag scores to give you an idea
#    of how well your forest is performing.

# 5. Feature importance: The forest keeps track of which features are most useful for making predictions.

# This implementation is great for learning about Random Forests or when you need a regressor
# that you can tweak and understand. It's built on top of a custom Decision Tree Regressor,
# so you have control all the way down to how individual trees are grown.

# Flattered you've made it this far!
