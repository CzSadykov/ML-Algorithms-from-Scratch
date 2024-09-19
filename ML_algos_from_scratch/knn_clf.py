import pandas as pd
import numpy as np


class MyKNNClf:
    """
    A friendly implementation of K-Nearest Neighbors Classifier.

    This class provides functionality to perform classification using the K-Nearest Neighbors algorithm.
    It supports various distance metrics and weighting schemes for prediction.

    Attributes:
        k (int): Number of neighbors to use for classification.
        metric (str): Distance metric to use ('euclidean', 'manhattan', 'chebyshev', or 'cosine').
        weight (str): Weighting scheme for neighbors ('uniform', 'distance', or 'rank').
        train_size (int): Number of samples in the training set.
        X_train (pd.DataFrame): Features of the training set.
        y_train (pd.Series): Labels of the training set.
    """

    def __init__(
            self,
            k: int = 5,
            metric: str = 'euclidean',
            weight: str = 'uniform'
            ):
        """
        Initialize the KNN Classifier.

        Args:
            k (int): Number of neighbors to use. Defaults to 5.
            metric (str): Distance metric to use. Defaults to 'euclidean'.
            weight (str): Weighting scheme for neighbors. Defaults to 'uniform'.
        """
        self.k = k
        self.metric = metric
        self.weight = weight
        self.train_size = 0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the KNN Classifier to the training data.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels.

        Returns:
            self: The fitted classifier.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.train_size = X.shape
        return self

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

    def _calc_distance(self, row):
        """
        Calculate distances between a sample and all training samples using the chosen metric.

        Args:
            row (pd.Series): A single sample to calculate distances for.

        Returns:
            np.array: Distances to all training samples.
        """
        if self.metric == 'euclidean':
            distance = np.sum(
                (self.X_train.values - row.values) ** 2, axis=1
                ) ** 0.5
        elif self.metric == 'manhattan':
            distance = np.sum(np.abs(self.X_train.values - row.values), axis=1)
        elif self.metric == 'chebyshev':
            distance = np.max(np.abs(self.X_train.values - row.values), axis=1)
        elif self.metric == 'cosine':
            distance = 1 - (
                np.sum(self.X_train.values * row.values, axis=1)
                / (np.sum(self.X_train.values ** 2, axis=1) ** 0.5
                   * (np.sum(row.values ** 2)) ** 0.5)
                )
        return distance

    def _calc_proba(self, row):
        """
        Calculate the probability of a sample belonging to the positive class.

        Args:
            row (pd.Series): A single sample to calculate probability for.

        Returns:
            float: Probability of the sample belonging to the positive class.
        """
        distances = self._calc_distance(row)
        sorted_indices = np.argsort(distances)[:self.k]
        nearest_neighbors = self.y_train.iloc[sorted_indices].values
        weights = self._calc_weights(distances[sorted_indices])

        class_weights = np.bincount(
            nearest_neighbors, weights=weights, minlength=2
            )
        return class_weights[1] / np.sum(class_weights)

    def predict(self, X: pd.DataFrame):
        """
        Predict class labels for samples in X.

        Args:
            X (pd.DataFrame): Samples to predict.

        Returns:
            pd.Series: Predicted class labels.
        """
        return X.apply(
            lambda row: (self._calc_proba(row) > 0.5).astype(int), axis=1
            )

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict class probabilities for samples in X.

        Args:
            X (pd.DataFrame): Samples to predict probabilities for.

        Returns:
            pd.Series: Predicted probabilities of belonging to the positive class.
        """
        return X.apply(self._calc_proba, axis=1)

    def __repr__(self):
        """
        Return a string representation of the MyKNNClf object.

        Returns:
            str: A string containing the class name and its parameters.
        """
        atts = ', '.join([f'{k}={v}' for k, v in vars(self).items()])
        return f'MyKNNClf class: {atts}'

# Description of the class and its functionality:

# MyKNNClf is a beginner-friendly implementation of a K-Nearest Neighbors Classifier. It's designed to be
# easy to use while still offering a good degree of customization. Here's what it can do:

# 1. Create a classifier: You can specify the number of neighbors (k), the distance metric to use,
#    and the weighting scheme for the neighbors.

# 2. Train the classifier: Use the 'fit' method to store your training data. KNN is a lazy learner,
#    so it doesn't actually build a model, it just remembers the training data.

# 3. Make predictions: Once trained, you can use 'predict' to classify new data, or 'predict_proba'
#    to get class probabilities.

# 4. Flexibility in distance metrics: You can choose between Euclidean, Manhattan, Chebyshev, or Cosine distance.

# 5. Customizable weighting: You can use uniform weights, distance-based weights, or rank-based weights for prediction.

# This implementation is great for learning about KNN or when you need a classifier that you can easily
# understand and tweak. It's built to work with pandas DataFrames and Series, making it convenient to use
# with typical data science workflows.

# Flatter you've made it this far!
