import pandas as pd
import numpy as np
import random
from ML_algos_from_scratch.decision_tree_clf import MyTreeClf


class MyForestClf:
    """
    This class allows you to create, train, and use a Random Forest for classification tasks.
    It's built on top of our custom Decision Tree Classifier and offers various customization options.

    Attributes:
        n_estimators (int): Number of trees in the forest.
        max_features (float): Fraction of features to consider for each tree.
        max_samples (float): Fraction of samples to use for each tree.
        random_state (int): Seed for random number generation.
        max_depth (int): Maximum depth of each tree.
        min_samples_split (int): Minimum samples required to split an internal node.
        max_leafs (int): Maximum number of leaf nodes allowed in each tree.
        bins (int): Number of bins for continuous features.
        criterion (str): The function to measure the quality of a split.
        oob_score (str): Type of out-of-bag score to compute.
        oob_score_ (float): Computed out-of-bag score.
        leafs_cnt (int): Total number of leaf nodes in the forest.
        fi (dict): Feature importances.
        forest (list): List of decision trees in the forest.
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
            criterion: str = 'entropy',
            oob_score: str = None
            ):
        """
        Initialize the Random Forest Classifier.

        Args:
            n_estimators (int): Number of trees in the forest.
            max_features (float): Fraction of features to consider for each tree.
            max_samples (float): Fraction of samples to use for each tree.
            random_state (int): Seed for random number generation.
            max_depth (int): Maximum depth of each tree.
            min_samples_split (int): Minimum samples required to split an internal node.
            max_leafs (int): Maximum number of leaf nodes allowed in each tree.
            bins (int): Number of bins for continuous features.
            criterion (str): The function to measure the quality of a split.
            oob_score (str): Type of out-of-bag score to compute.
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
        self.criterion = criterion
        self.random_state = random_state
        self.leafs_cnt = 0
        self.fi = {}

        if oob_score in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            self.oob_score = oob_score
        else:
            raise ValueError('Invalid oob_score value')

        self.oob_score_ = None

    def _calc_metric(self, y_true: pd.Series, y_pred: pd.Series):
        """
        Calculate the specified out-of-bag metric.

        Args:
            y_true (pd.Series): True labels.
            y_pred (pd.Series): Predicted probabilities.

        Returns:
            float: Calculated metric value.
        """
        if self.oob_score != 'roc_auc':
            y_pred = (y_pred > 0.5).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        if self.oob_score == 'accuracy':
            return (tp + tn) / (tp + fp + fn + tn)

        if self.oob_score == 'precision':
            return tp / (tp + fp) if (tp + fp) != 0 else 0.0

        if self.oob_score == 'recall':
            return tp / (tp + fn) if (tp + fn) != 0 else 0.0

        if self.oob_score == 'f1':
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
            return 2 * (precision * recall) / (precision + recall)

        if self.oob_score == 'roc_auc':
            positives = np.sum(y_true == 1)
            negatives = np.sum(y_true == 0)

            y_prob = np.round(y_pred, 10)

            sorted_idx = np.argsort(-y_prob)
            y_sorted = y_true[sorted_idx]
            y_prob_sorted = y_prob[sorted_idx]

            roc_auc_score = 0

            for prob, pred in zip(y_prob_sorted, y_sorted):
                if pred == 0:
                    roc_auc_score += (
                        np.sum(y_sorted[y_prob_sorted > prob])
                        + np.sum(y_sorted[y_prob_sorted == prob]) / 2
                        )

            roc_auc_score /= positives * negatives

            return roc_auc_score

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Build a forest of trees from the training set (X, y).

        Args:
            X (pd.DataFrame): The input samples.
            y (pd.Series): The target values.
        """
        self.forest = []
        rows, cols = X.shape
        self.leafs_cnt = 0
        random.seed(self.random_state)

        feature_list = list(X.columns)
        self.fi = {col: 0 for col in feature_list}

        init_cols = int(np.round(cols * self.max_features))
        init_rows = int(np.round(rows * self.max_samples))

        if self.oob_score:
            self.oob_predictions_ = {}

        for _ in range(self.n_estimators):
            cols_sample = random.sample(
                feature_list, init_cols
                )
            rows_sample = random.sample(
                range(rows), init_rows
                )
            X_sample = X.iloc[rows_sample][cols_sample]
            y_sample = y.iloc[rows_sample]

            tree = MyTreeClf(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_leafs=self.max_leafs,
                bins=self.bins,
                criterion=self.criterion
            )
            tree.fit(X_sample, y_sample, len(y))
            self.forest.append(tree)

            for col in cols_sample:
                self.fi[col] += tree.fi[col]

            self.leafs_cnt += tree.leafs_cnt

            if self.oob_score:
                oob_X = X.iloc[~X.index.isin(rows_sample)][cols_sample]
                oob_pred = tree.predict_proba(oob_X)
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
            self.oob_score_ = self._calc_metric(oob_y, np.array(oob_preds))

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Args:
            X (pd.DataFrame): The input samples.

        Returns:
            np.array: The class probabilities of the input samples.
        """
        return np.array(
            [list(tree.predict_proba(X)) for tree in self.forest]
            ).mean(axis=0)

    def predict(self, X, type):
        """
        Predict class labels for X.

        Args:
            X (pd.DataFrame): The input samples.
            type (str): The type of prediction ('mean' or 'vote').

        Returns:
            np.array: The predicted class labels.
        """
        pred_probs = np.array(
            [list(tree.predict_proba(X)) for tree in self.forest]
            )
        if type == 'mean':
            pred = pred_probs.mean(axis=0)
            pred = (pred > 0.5).astype(int)
        elif type == 'vote':
            pred_votes = np.apply_along_axis(
                lambda x: np.bincount(x),
                axis=0, arr=(pred_probs > 0.5).astype(int)
            )
            pred = pred_votes.argmax(axis=0)
        return pred

    def __repr__(self):
        """
        Return a string representation of the MyForestClf object.

        Returns:
            str: A string containing the class name and its parameters.
        """
        atts = ', '.join([f'{k}={v}' for k, v in vars(self).items()])
        return f'MyForestClf class: {atts}'

# Description of the class and its functionality:

# MyForestClf is a beginner-friendly implementation of a Random Forest Classifier. It's designed to be
# easy to use while still offering a good degree of customization. Here's what it can do:

# 1. Create a forest: You can specify the number of trees, how many features and samples to use
#    for each tree, and various other parameters to control tree growth.

# 2. Train the forest: Use the 'fit' method to build your forest from training data. It'll create
#    multiple decision trees, each trained on a random subset of your data and features.

# 3. Make predictions: Once trained, you can use 'predict' to classify new data, or 'predict_proba'
#    to get class probabilities.

# 4. Evaluate performance: If you choose, it can calculate out-of-bag scores to give you an idea
#    of how well your forest is performing.

# 5. Feature importance: The forest keeps track of which features are most useful for making decisions.

# This implementation is great for learning about Random Forests or when you need a classifier
# that you can tweak and understand. It's built on top of our custom Decision Tree Classifier,
# so you have control all the way down to how individual trees are grown.

# Enjoy exploring and experimenting with your very own Random Forest!
# I'm flattered you've made it this far!