# ML Algorithms from Scratch

This repository contains implementations of various machine learning algorithms from scratch. These implementations are designed to be beginner-friendly and educational (considering that author is a beginner himself lol), providing insights into the inner workings of popular ML algorithms.

## Modules

### 1. Linear Regression (lin_reg.py)

The `MyLineReg` class implements linear regression with the following features:
- Gradient descent optimization
- Multiple evaluation metrics (MAE, MAPE, R2, MSE, RMSE)
- Regularization options (L1, L2, Elastic Net)
- Stochastic Gradient Descent (SGD) support
- Customizable learning rate

### 2. Logistic Regression (log_reg.py)

The `MyLogReg` class provides a logistic regression implementation with:
- Gradient descent optimization
- Various regularization techniques (L1, L2, Elastic Net)
- Multiple evaluation metrics
- Stochastic Gradient Descent option
- Flexible learning rate (fixed or callable)

### 3. K-Nearest Neighbors Classifier (knn_clf.py)

The `MyKNNClf` class implements a K-Nearest Neighbors classifier with:
- Support for various distance metrics (Euclidean, Manhattan, Chebyshev, Cosine)
- Customizable weighting schemes (uniform, distance-based, rank-based)
- Methods for both classification and probability estimation

### 4. K-Nearest Neighbors Regressor (knn_reg.py)

The `MyKNNReg` class provides a K-Nearest Neighbors regressor with similar features to the classifier version, adapted for regression tasks.

### 5. Decision Tree Classifier (decision_tree_clf.py)

The `MyTreeClf` class implements a decision tree classifier with:
- Customizable tree growth parameters (max depth, min samples to split, max leaves)
- Support for both entropy and gini impurity as split criteria
- Feature importance calculation

### 6. Decision Tree Regressor (decision_tree_reg.py)

The `MyTreeReg` class provides a decision tree regressor with similar features to the classifier version, adapted for regression tasks.

### 7. Random Forest Classifier (random_forest_clf.py)

The `MyForestClf` class implements a random forest classifier with:
- Customizable forest-level parameters (number of trees, feature and sample ratios)
- Individual tree customization options
- Out-of-bag score calculation
- Feature importance aggregation

### 8. Random Forest Regressor (random_forest_reg.py)

The `MyForestReg` class provides a random forest regressor with similar features to the classifier version, adapted for regression tasks.

### 9. Principal Component Analysis (pca.py)

The `MyPCA` class implements Principal Component Analysis for dimensionality reduction, including:
- Customizable number of components
- Data centering and covariance matrix computation
- Eigenvalue decomposition for finding principal components

## Usage

Each module can be imported and used independently. Here's a general pattern for using these algorithms:

```python
from lin_reg import MyLineReg
from log_reg import MyLogReg
from knn_clf import MyKNNClf
from knn_reg import MyKNNReg
from decision_tree_clf import MyTreeClf
from decision_tree_reg import MyTreeReg
from random_forest_clf import MyForestClf
from random_forest_reg import MyForestReg
from pca import MyPCA
```

## Contributing

This is a personal project for educational purposes, and contributions are not expected. However, if you have suggestions or improvements, feel free to open an issue or submit a pull request.

## License

You can use the code however you want, but please don't expect any support if you run into issues...