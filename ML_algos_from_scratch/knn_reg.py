import numpy as np
import pandas as pd


class MyKNNReg:
    """
    A beginner-friendly implementation of K-Nearest Neighbors Regressor.

    This class provides functionality to perform regression using the K-Nearest Neighbors algorithm.
    It supports various distance metrics and weighting schemes for prediction.

    Attributes:
        k (int): Number of neighbors to use for regression.
        metric (str): Distance metric to use ('euclidean', 'manhattan', 'chebyshev', or 'cosine').
        weight (str): Weighting scheme for neighbors ('uniform', 'distance', or 'rank').
        train_size (int): Number of samples in the training set.
        X_train (pd.DataFrame): Features of the training set.
        y_train (pd.Series): Target values of the training set.
    """

    def __init__(
            self,
            k: int = 3,
            metric: str = 'euclidean',
            weight: str = 'uniform'
            ):
        """
        Initialize the KNN Regressor.

        Args:
            k (int): Number of neighbors to use. Defaults to 3.
            metric (str): Distance metric to use. Defaults to 'euclidean'.
            weight (str): Weighting scheme for neighbors. Defaults to 'uniform'.
        """
        self.k = k
        self.metric = metric
        self.weight = weight
        self.train_size = 0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the KNN Regressor to the training data.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target values.

        Returns:
            self: The fitted regressor.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.train_size = X.shape
        return self

    def _calc_distance(self, row):
        """
        Calculate distances between a single sample and all training samples.

        Args:
            row (pd.Series): A single sample to calculate distances for.

        Returns:
            np.array: Distances to all training samples.
        """
        if self.metric == 'euclidean':
            return np.sum(
                (self.X_train.values - row.values) ** 2, axis=1
                ) ** 0.5
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train.values - row.values), axis=1)
        elif self.metric == 'chebyshev':
            return np.max(np.abs(self.X_train.values - row.values), axis=1)
        elif self.metric == 'cosine':
            return 1 - (
                np.sum(self.X_train.values * row.values, axis=1)
                / (np.sum(self.X_train.values ** 2, axis=1) ** 0.5
                   * (np.sum(row.values ** 2)) ** 0.5)
                )

    def _calc_weights(self, distances):
        """
        Calculate weights for the nearest neighbors based on the chosen weighting scheme.

        Args:
            distances (np.array): Distances to the nearest neighbors.

        Returns:
            np.array: Weights for the nearest neighbors.
        """
        if self.weight == 'uniform':
            return np.ones(self.k)
        elif self.weight == 'distance':
            return 1 / (distances + 1e-20)
        elif self.weight == 'rank':
            return 1 / (np.arange(self.k) + 1)

    def _calc_means(self, row):
        """
        Calculate the weighted mean of target values for the k nearest neighbors.

        Args:
            row (pd.Series): A single sample to calculate the mean for.

        Returns:
            float: Predicted target value for the sample.
        """
        distances = self._calc_distance(row)
        sorted_indices = np.argsort(distances)[:self.k]
        targets = self.y_train.iloc[sorted_indices].values
        weights = self._calc_weights(distances[sorted_indices])
        return np.sum(targets * weights) / np.sum(weights)

    def predict(self, X: pd.DataFrame):
        """
        Predict target values for samples in X.

        Args:
            X (pd.DataFrame): Samples to predict target values for.

        Returns:
            pd.Series: Predicted target values.
        """
        return X.apply(self._calc_means, axis=1)

    def __repr__(self):
        """
        Return a string representation of the MyKNNReg object.

        Returns:
            str: A string containing the class name and its parameters.
        """
        atts = ', '.join([f'{k}={v}' for k, v in vars(self).items()])
        return f'MyKNNReg class: {atts}'

# Description of the class and its functionality:

# MyKNNReg is a beginner-friendly implementation of a K-Nearest Neighbors Regressor. It's designed to be
# easy to use while still offering a good degree of customization. Here's what it can do:

# 1. Create a regressor: You can specify the number of neighbors (k), the distance metric to use,
#    and the weighting scheme for the neighbors.

# 2. Train the regressor: Use the 'fit' method to store your training data. KNN is a lazy learner,
#    so it doesn't actually build a model, it just remembers the training data.

# 3. Make predictions: Once trained, you can use 'predict' to estimate target values for new data.

# 4. Flexibility in distance metrics: You can choose between Euclidean, Manhattan, Chebyshev, or Cosine distance.

# 5. Customizable weighting: You can use uniform weights, distance-based weights, or rank-based weights for prediction.

# This implementation is great for learning about KNN or when you need a regressor that you can easily
# understand and tweak. It's built to work with pandas DataFrames and Series, making it convenient to use
# with typical data science workflows.
