import numpy as np
import pandas as pd


class MyPCA:
    """
    A beginner-friendly implementation of Principal Component Analysis (PCA).

    This class provides functionality to perform PCA on a given dataset.
    PCA is a dimensionality reduction technique that finds the principal
    components of the data, which are the directions of maximum variance.

    Attributes:
        n_components (int): Number of principal components to compute.
        components (np.ndarray): Principal components (eigenvectors) of the data.
    """

    def __init__(self, n_components: int = 3):
        """
        Initialize the PCA object.

        Args:
            n_components (int): Number of principal components to compute. Defaults to 3.
        """
        self.n_components = n_components

    def fit_transform(self, X: pd.DataFrame):
        """
        Fit the PCA model to the data and transform it.

        This method centers the data, computes the covariance matrix,
        finds the eigenvectors and eigenvalues, and projects the data
        onto the principal components.

        Args:
            X (pd.DataFrame): Input data to transform.

        Returns:
            np.ndarray: Transformed data (projected onto principal components).
        """
        X_centered = X - X.mean(axis=0)
        cov_matrix = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]
        return X_centered @ self.components

    def __repr__(self):
        """
        Return a string representation of the MyPCA object.

        Returns:
            str: String representation of the object.
        """
        atts = ', '.join(f'{k}={v}' for k, v in vars(self).items())
        return f'MyPCA class: {atts}'

# Description of the class and its functionality:
#
# MyPCA is a beginner-friendly implementation of Principal Component Analysis (PCA).
# It's designed to be easy to use while still providing the core functionality of PCA.
# Here's what it can do:
#
# 1. Initialize a PCA object: You can specify the number of principal components to compute.
#
# 2. Fit and transform data: The 'fit_transform' method performs PCA on the input data and
#    returns the transformed data projected onto the principal components.
#
# 3. Dimensionality reduction: By specifying fewer components than the original number of
#    features, you can reduce the dimensionality of your data while preserving most of the variance.
#
# 4. Compute principal components: The class computes and stores the principal components
#    (eigenvectors) of the data.
#
# This implementation is great for learning about PCA or when you need a simple PCA
# implementation that you can easily understand and modify. It's built to work with
# pandas DataFrames, making it convenient to use in typical data science workflows. 
