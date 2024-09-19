import numpy as np
import pandas as pd
import random
from typing import Union, Callable

class MyLineReg:
    """
    A beginner-friendly linear regression class.

    This class implements linear regression with various features:
    - Gradient descent optimization
    - Multiple evaluation metrics
    - Regularization options (L1, L2, Elastic Net)
    - Stochastic Gradient Descent (SGD) support
    - Customizable learning rate
    """

    @staticmethod
    def mae(y_pred: pd.Series, y_true: pd.Series):
        """
        Calculate Mean Absolute Error (MAE) between predicted and true values.

        Args:
            y_pred (pd.Series): Predicted values
            y_true (pd.Series): True values

        Returns:
            float: Mean Absolute Error
        """
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mape(y_pred: pd.Series, y_true: pd.Series):
        """
        Calculate Mean Absolute Percentage Error (MAPE) between predicted and true values.

        Args:
            y_pred (pd.Series): Predicted values
            y_true (pd.Series): True values

        Returns:
            float: Mean Absolute Percentage Error
        """
        return np.mean(np.abs((y_pred - y_true)/y_true)) * 100

    @staticmethod
    def r2(y_pred: pd.Series, y_true: pd.Series):
        """
        Calculate R-squared (coefficient of determination) between predicted and true values.

        Args:
            y_pred (pd.Series): Predicted values
            y_true (pd.Series): True values

        Returns:
            float: R-squared value
        """
        return 1 - np.mean((y_true - y_pred)**2/np.var(y_true))

    @staticmethod
    def mse(y_pred: pd.Series, y_true: pd.Series):
        """
        Calculate Mean Squared Error (MSE) between predicted and true values.

        Args:
            y_pred (pd.Series): Predicted values
            y_true (pd.Series): True values

        Returns:
            float: Mean Squared Error
        """
        return np.mean((y_pred - y_true)**2)

    @staticmethod
    def rmse(y_pred: pd.Series, y_true: pd.Series):
        """
        Calculate Root Mean Squared Error (RMSE) between predicted and true values.

        Args:
            y_pred (pd.Series): Predicted values
            y_true (pd.Series): True values

        Returns:
            float: Root Mean Squared Error
        """
        return np.sqrt(np.mean((y_pred - y_true)**2))

    @staticmethod
    def get_metric_score(y_pred: pd.Series, y_true: pd.Series, metric: str):
        """
        Calculate the score for a specified metric.

        Args:
            y_pred (pd.Series): Predicted values
            y_true (pd.Series): True values
            metric (str): Name of the metric to calculate

        Returns:
            float: Score for the specified metric
        """
        return eval(f'MyLineReg.{metric}(y_pred, y_true)')

    def __init__(
        self,
        n_iter: int = 100,
        learning_rate: Union[float, Callable] = 0.01,
        metric: str = None,
        reg: str = None,
        l1_coef: float = 0,
        l2_coef: float = 0,
        sgd_sample: Union[int, float] = None,
        random_state: int = 42
    ):
        """
        Initialize the MyLineReg class with specified parameters.

        Args:
            n_iter (int): Number of iterations for gradient descent
            learning_rate (float or Callable): Learning rate or a function to calculate it
            metric (str): Metric to use for evaluation
            reg (str): Type of regularization ('l1', 'l2', or 'elasticnet')
            l1_coef (float): L1 regularization coefficient
            l2_coef (float): L2 regularization coefficient
            sgd_sample (int or float): Sample size for Stochastic Gradient Descent
            random_state (int): Random seed for reproducibility
        """
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.metric_value = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Calculate the loss (cost) function value.

        Args:
            y_pred (np.ndarray): Predicted values
            y_true (np.ndarray): True values

        Returns:
            float: Loss value
        """
        loss = np.mean((y_true - y_pred)**2)
        if self.reg == 'l1':
            loss += self.l1_coef * np.sum(np.abs(self.weights))
        elif self.reg == 'l2':
            loss += self.l2_coef * np.sum(self.weights ** 2)
        elif self.reg == 'elasticnet':
            loss += (
                self.l1_coef * np.sum(np.abs(self.weights))
                + self.l2_coef * np.sum(self.weights ** 2)
            )
        return loss

    def gradient(
            self,
            X: np.ndarray,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            batch_idx: list
            ):
        """
        Calculate the gradient of the loss function.

        Args:
            X (np.ndarray): Input features
            y_pred (np.ndarray): Predicted values
            y_true (np.ndarray): True values
            batch_idx (list): Indices of the current batch

        Returns:
            np.ndarray: Gradient of the loss function
        """
        X = X[batch_idx]
        y_true = y_true[batch_idx].reshape(-1, 1)
        y_pred = y_pred[batch_idx].reshape(-1, 1)

        grad = 2 / X.shape[0] * X.T @ (y_pred - y_true)
        if self.reg == 'l1':
            grad += self.l1_coef * np.sign(self.weights)
        elif self.reg == 'l2':
            grad += self.l2_coef * 2 * self.weights
        elif self.reg == 'elasticnet':
            grad += (
                self.l1_coef * np.sign(self.weights)
                + self.l2_coef * 2 * self.weights
            )
        return grad

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        """
        Fit the linear regression model to the training data.

        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target values
            verbose (bool): If True, print progress during training

        Returns:
            None
        """
        y_true = y.values
        features = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        self.weights = np.ones((features.shape[1], 1))

        random.seed(self.random_state)

        for i in range(1, self.n_iter+1):

            if self.sgd_sample:
                if 0 < self.sgd_sample < 1:
                    sample_size = int(self.sgd_sample * X.shape[0])
                else:
                    sample_size = self.sgd_sample
                batch_idx = random.sample(
                    range(features.shape[0]), sample_size
                    )
            else:
                batch_idx = list(range(features.shape[0]))

            y_pred = np.dot(features, self.weights)
            loss = self.loss(y_pred, y_true)
            grad = self.gradient(features, y_pred, y_true, batch_idx)

            if callable(self.learning_rate):
                cur_learning_rate = self.learning_rate(i)
            else:
                cur_learning_rate = self.learning_rate

            self.weights -= cur_learning_rate * grad

            if self.metric:
                self.metric_value = self.get_metric_score(
                    y_pred, y_true, self.metric
                    )

            if verbose:
                if (i == 1) or (i % verbose == 0):
                    log = f'{i} | loss: {loss}'
                    if self.metric:
                        log += f' | metric: {MyLineReg.get_metric_score(y_pred, y_true, self.metric)}'
                    print(log)

    def get_coef(self):
        """
        Get the coefficients (weights) of the linear regression model.

        Returns:
            np.ndarray: Model coefficients (excluding the intercept)
        """
        return self.weights[1:]

    def predict(self, X: pd.DataFrame):
        """
        Make predictions using the trained model.

        Args:
            X (pd.DataFrame): Input features

        Returns:
            np.ndarray: Predicted values
        """
        features = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        ypred = features @ self.weights
        return ypred

    def get_best_score(self):
        """
        Get the best score achieved during training for the specified metric.

        Returns:
            float: Best score (or None if no metric was specified)
        """
        return self.metric_value if self.metric else None

    def __str__(self):
        """
        Return a string representation of the MyLineReg object.

        Returns:
            str: String representation of the object
        """
        attributes = ', '.join(
            f'{key}={value}' for key, value in vars(self).items()
            )
        return f'MyLineReg class: {attributes}'

# Description of the class and its functionality:
#
# MyLineReg is a beginner-friendly implementation of a Linear Regression model. It's designed to be
# easy to use while still offering a good degree of customization. Here's what it can do:
#
# 1. Create a regressor: You can specify the number of iterations, learning rate, evaluation metric,
#    regularization type and parameters, and options for Stochastic Gradient Descent.
#
# 2. Train the regressor: Use the 'fit' method to train the model on your data. It uses gradient descent
#    for optimization and supports various regularization techniques.
#
# 3. Make predictions: Once trained, you can use 'predict' to estimate target values for new data.
#
# 4. Evaluate performance: The class provides multiple evaluation metrics (MAE, MAPE, R2, MSE, RMSE)
#    and allows you to track a specified metric during training.
#
# 5. Regularization options: You can use L1, L2, or Elastic Net regularization to prevent overfitting.
#
# 6. Stochastic Gradient Descent: The class supports SGD for faster training on large datasets.
#
# 7. Customizable learning rate: You can use a fixed learning rate or provide a function for dynamic rates.
#
# This implementation is great for learning about linear regression or when you need a regressor that you can
# easily understand and modify. It's built to work with pandas DataFrames and Series, making it convenient
# to use with typical data science workflows.
# Glad you've made it this far!
