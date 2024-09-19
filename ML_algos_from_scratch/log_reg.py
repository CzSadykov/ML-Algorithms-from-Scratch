import numpy as np
import pandas as pd
import random


class MyLogReg:
    """
    A beginner-friendly implementation of Logistic Regression.

    This class provides functionality to create and use a Logistic Regression model for binary classification tasks.
    It supports various regularization techniques and optimization methods.

    Attributes:
        n_iter (int): Number of iterations for training.
        learning_rate (float or callable): Learning rate for gradient descent.
        weights (numpy.ndarray): Model coefficients.
        metric (str): Metric to evaluate model performance.
        score (float): Best score achieved during training.
        random_state (int): Seed for random number generation.
        reg (str): Type of regularization (None, 'l1', 'l2', or 'elasticnet').
        l1_coef (float): L1 regularization coefficient.
        l2_coef (float): L2 regularization coefficient.
        sgd_sample (int or float): Number or fraction of samples to use in stochastic gradient descent.
    """

    def __init__(self,
                 n_iter: int = 10,
                 learning_rate=0.1,
                 weights=None,
                 metric: str = None,
                 reg: str = None,
                 l1_coef: float = 0,
                 l2_coef: float = 0,
                 sgd_sample=None,
                 random_state: int = 42
                 ):
        """
        Initialize the Logistic Regression model.

        Args:
            n_iter (int): Number of iterations for training. Defaults to 10.
            learning_rate (float or callable): Learning rate for gradient descent. Defaults to 0.1.
            weights (numpy.ndarray, optional): Initial weights for the model. Defaults to None.
            metric (str, optional): Metric to evaluate model performance. Defaults to None.
            reg (str, optional): Type of regularization. Defaults to None.
            l1_coef (float): L1 regularization coefficient. Defaults to 0.
            l2_coef (float): L2 regularization coefficient. Defaults to 0.
            sgd_sample (int or float, optional): Number or fraction of samples to use in SGD. Defaults to None.
            random_state (int): Seed for random number generation. Defaults to 42.
        """
        self.n_iter = n_iter
        self.weights = weights
        self.metric = metric
        self.score = None
        
        if not isinstance(random_state, int):
            raise ValueError("Error! Choose a valid integer for random_state")
        self.random_state = random_state
        
        if not callable(learning_rate):
            if 0.00001 <= learning_rate <= 100:
                self.learning_rate = learning_rate
            else:
                raise ValueError("Error! Learning rate must be between [1e-5, 100]")
        else:
            self.learning_rate = learning_rate
        
        valid_regs = [None, 'l1', 'l2', 'elasticnet']
        if reg not in valid_regs:
            raise ValueError("Error! Choose reg from None, 'l1', 'l2' and 'elasticnet'")
        self.reg = reg   
        
        if 0 <= l1_coef <= 1:
            self.l1_coef = l1_coef
        else:
            raise ValueError("Error! l1_coef must be a float between 0 and 1.")
            
        if 0 <= l2_coef <= 1:
            self.l2_coef = l2_coef
        else:
            raise ValueError("Error! l2_coef must be a float between 0 and 1.")
            
        if isinstance(sgd_sample, (int, float)):
            if sgd_sample < 0:
                raise ValueError("Error! sgd_sample must be positive")
            self.sgd_sample = sgd_sample
        else:
            self.sgd_sample = None

    def __repr__(self):
        """
        Return a string representation of the MyLogReg object.

        Returns:
            str: A string containing the class name and its parameters.
        """
        atts = ', '.join(f'{k}={v}' for k, v in vars(self).items())
        return f'MyLogReg class: {atts}'

    def _loss(self, X: pd.DataFrame, y_pred: np.ndarray, y_true: pd.Series):
        """
        Calculate the loss function value.

        Args:
            X (pd.DataFrame): Input features.
            y_pred (np.ndarray): Predicted probabilities.
            y_true (pd.Series): True labels.

        Returns:
            float: The calculated loss value.
        """
        loss = -1/X.shape[0] * np.sum((y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
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

    def _grad(self, X: np.ndarray, y_pred: np.ndarray, y_true: pd.Series, batch_idx):
        """
        Calculate the gradient of the loss function.

        Args:
            X (np.ndarray): Input features.
            y_pred (np.ndarray): Predicted probabilities.
            y_true (pd.Series): True labels.
            batch_idx (list): Indices of the current batch.

        Returns:
            np.ndarray: The calculated gradient.
        """
        X = X[batch_idx]
        y_pred = y_pred[batch_idx]
        y_true = y_true[batch_idx]
        
        grad = 1/X.shape[0] * np.dot(X.T, (y_pred - y_true))
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
    
    def _calc_metric(self, X: pd.DataFrame, y_true: pd.Series):
        """
        Calculate the specified evaluation metric.

        Args:
            X (pd.DataFrame): Input features.
            y_true (pd.Series): True labels.

        Returns:
            float: The calculated metric value.
        """
        preds = self.predict(X)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))
        
        if self.metric == 'accuracy':
            return (tp + tn) / (tp + fp + fn + tn)

        if self.metric == 'precision':
            return tp / (tp + fp) if (tp + fp) != 0 else 0.0

        if self.metric == 'recall':
            return tp / (tp + fn) if (tp + fn) != 0 else 0.0

        if self.metric == 'f1':
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

        if self.metric == 'roc_auc':
            preds = self.predict_proba(X)
            idx = np.lexsort((y_true, np.round(preds, 10)))[::-1]
            preds, labels = preds[idx], y_true[idx]
            pairs = 1
            ones, total = 0, 0
            prev = None
            
            for p, l in zip(preds, labels):
                if l == 1 and p != prev:
                    ones += 1
                    prev = p
                elif l == 1 and p == prev:
                    ones += 1
                    pairs += 1
                elif l == 0 and p == prev:
                    total += ones - (pairs / 2)
                elif l == 0 and p != prev:
                    pairs = 1
                    total += ones
                    
            zeros = len(preds) - ones
            return total / (ones * zeros) if ones * zeros > 0 else 0.0  

    def fit(self, X: pd.DataFrame, y_true, verbose: int = False):
        """
        Fit the logistic regression model to the training data.

        Args:
            X (pd.DataFrame): Training features.
            y_true (pd.Series or np.ndarray): Training labels.
            verbose (int): If non-zero, print training progress every 'verbose' iterations.

        Returns:
            self: The fitted model.
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        features = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        self.weights = np.ones(features.shape[1])
        
        random.seed(self.random_state)
        
        for i in range(1, self.n_iter + 1):
            
            if self.sgd_sample:
                if 0 < self.sgd_sample < 1:
                    sample_size = int(self.sgd_sample * features.shape[0])
                else:
                    sample_size = self.sgd_sample
                batch_idx = random.sample(range(features.shape[0]), sample_size)
            else:
                batch_idx = list(range(features.shape[0]))

            y_pred = 1 / (1 + np.exp(-np.dot(features, self.weights)))
            loss = self._loss(features, y_pred, y_true)
            grad = self._grad(features, y_pred, y_true, batch_idx)

            cur_learning_rate = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate

            self.weights -= cur_learning_rate * grad
            self.score = self._calc_metric(X, y_true)

            if verbose and (i == 1 or i % verbose == 0):
                log = f'{i} | learning_rate: {cur_learning_rate} | loss: {loss}'
                if self.metric:
                    log += f' | {self.metric}: {self.score}'
                print(log)

        return self
    
    def predict_proba(self, X: pd.DataFrame):
        """
        Predict class probabilities for the given features.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted probabilities for the positive class.
        """
        features = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return 1 / (1 + np.exp(-np.dot(features, self.weights)))

    def predict(self, X: pd.DataFrame):
        """
        Predict class labels for the given features.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted class labels (0 or 1).
        """
        preds = self.predict_proba(X)
        return np.where(preds > 0.5, 1, 0)

    def get_coef(self):
        """
        Get the coefficients (weights) of the model.

        Returns:
            np.ndarray: Model coefficients, excluding the intercept.
        """
        return self.weights[1:]

    def get_best_score(self):
        """
        Get the best score achieved during training.

        Returns:
            float: The best score according to the specified metric.
        """
        return self.score

# Description of the class and its functionality:

# MyLogReg is a beginner-friendly implementation of Logistic Regression for binary classification.
# It offers various features to help users understand and experiment with logistic regression:

# 1. Customizable training: Users can set the number of iterations, learning rate, and initial weights.
# 2. Regularization options: Supports L1, L2, and Elastic Net regularization to prevent overfitting.
# 3. Stochastic Gradient Descent: Allows training on subsets of data for faster computation on large datasets.
# 4. Multiple evaluation metrics: Supports accuracy, precision, recall, F1-score, and ROC AUC for model evaluation.
# 5. Flexible learning rate: Accepts both fixed values and callable learning rate schedules.
# 6. Verbose training: Option to print training progress, including loss and chosen metric.

# The class provides methods for:
# - Fitting the model to training data
# - Predicting probabilities and class labels for new data
# - Retrieving model coefficients and best achieved score

# This implementation is great for learning about logistic regression or when you need a customizable,
# transparent classifier. It's designed to be easy to use while still offering a good degree of flexibility.

# Enjoy exploring and experimenting with your very own Logistic Regression model!
