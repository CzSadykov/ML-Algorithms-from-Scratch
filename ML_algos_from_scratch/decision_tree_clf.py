import numpy as np
import pandas as pd


class Node:
    """
    A class representing a node in the decision tree.

    Attributes:
        feature (str): The feature used for splitting at this node.
        value_split (float): The threshold value for the split.
        value_leaf (float): The prediction value if this is a leaf node.
        side (str): Indicates whether this node is a 'left' or 'right' child.
        left (Node): The left child node.
        right (Node): The right child node.
    """
    def __init__(self):
        self.feature = None
        self.value_split = None
        self.value_leaf = None
        self.side = None
        self.left = None
        self.right = None


class MyTreeClf:
    """
    A custom implementation of a Decision Tree Classifier.

    This class provides functionality to build and use a decision tree for classification tasks.
    It supports various hyperparameters for tree construction and different split criteria.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        max_leafs (int): Maximum number of leaf nodes.
        bins (int): Number of bins for continuous features (if specified).
        criterion (str): The function to measure the quality of a split ('entropy' or 'gini').
    """

    def __init__(self,
                 max_depth=5,
                 min_samples_split=2,
                 max_leafs=20,
                 bins=None,
                 criterion='entropy'
                 ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1
        self.bins = bins
        self.__sum_tree_values = 0
        self.split_values = {}
        self.criterion = criterion
        self.fi = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, init_data_len: int = None):
        """
        Build a decision tree classifier from the training set (X, y).

        Args:
            X (pd.DataFrame): The input samples.
            y (pd.Series): The target values.
            init_data_len (int, optional): Initial data length for feature importance calculation.

        Returns:
            None
        """
        self.tree = None
        self.fi = {col: 0 for col in X.columns}

        def create_tree(root, X_root, y_root, side='root', depth=0):
            if root is None:
                root = Node()
            col_name, split_value, ig = self.get_best_split(X_root, y_root)

            proportion_ones = len(
                y_root[y_root == 1]
                ) / len(y_root) if len(y_root) else 0

            if (
                proportion_ones == 0
                or proportion_ones == 1
                or depth >= self.max_depth
                or len(y_root) < self.min_samples_split
                or (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs)
            ):
                root.side = side
                root.value_leaf = proportion_ones
                self.__sum_tree_values += root.value_leaf
                return root

            X_left = X_root.loc[X_root[col_name] <= split_value]
            y_left = y_root.loc[X_root[col_name] <= split_value]

            X_right = X_root.loc[X_root[col_name] > split_value]
            y_right = y_root.loc[X_root[col_name] > split_value]

            if len(X_left) == 0 or len(X_right) == 0:
                root.side = side
                root.value_leaf = proportion_ones
                self.__sum_tree_values += root.value_leaf
                return root

            if init_data_len is None:
                self.fi[col_name] += len(y_root) / len(y) * ig
            else:
                self.fi[col_name] += len(y_root) / init_data_len * ig

            root.feature = col_name
            root.value_split = split_value
            self.leafs_cnt += 1

            root.left = create_tree(
                root.left, X_left, y_left, 'left', depth + 1
                )
            root.right = create_tree(
                root.right, X_right, y_right, 'right', depth + 1
                )

            return root

        self.tree = create_tree(self.tree, X, y)

    def _predict_row(self, row):
        """
        Predict the class for a single input row.

        Args:
            row (pd.Series): A single input sample.

        Returns:
            float: The predicted class probability.
        """
        node = self.tree
        while node.feature is not None:
            if row[node.feature] <= node.value_split:
                node = node.left
            else:
                node = node.right
        return node.value_leaf

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict class probabilities for X.

        Args:
            X (pd.DataFrame): The input samples.

        Returns:
            pd.Series: The class probabilities of the input samples.
        """
        return X.apply(self._predict_row, axis=1)

    def predict(self, X: pd.DataFrame):
        """
        Predict class labels for X.

        Args:
            X (pd.DataFrame): The input samples.

        Returns:
            np.array: The predicted class labels.
        """
        return np.where(self.predict_proba(X) > 0.5, 1, 0)

    def print_tree(self, node=None, depth=0):
        """
        Print a text representation of the decision tree.

        Args:
            node (Node, optional): The current node to print. Defaults to the root.
            depth (int, optional): The current depth in the tree. Defaults to 0.

        Returns:
            None
        """
        if node is None:
            node = self.tree
        if node.feature is not None:
            print(f"{' ' * depth}{node.feature} > {node.value_split}")
            if node.left is not None:
                self.print_tree(node.left, depth + 1)
            if node.right is not None:
                self.print_tree(node.right, depth + 1)
        else:
            print(f"{' ' * depth}{node.side} = {node.value_leaf}")

    def get_best_split(self, X, y):
        """
        Find the best split for a node.

        Args:
            X (pd.DataFrame): The feature dataset.
            y (pd.Series): The target values.

        Returns:
            tuple: The best feature to split on, the split value, and the information gain.
        """
        count_labels = y.value_counts()
        p_zero = count_labels / count_labels.sum()
        s_zero = self.__node_rule(p_zero)

        col_name = None
        split_value = None
        s_cur_min = float('inf')

        for col in X.columns:
            if not (col in self.split_values.keys()):
                x_unique_values = np.unique(X[col])
                if (
                    self.bins is not None and
                    len(x_unique_values) - 1 >= self.bins
                ):
                    _, self.split_values[col] = np.histogram(
                     X[col], bins=self.bins
                    )
                    self.split_values[col] = self.split_values[col][1:-1]
                else:
                    self.split_values[col] = (
                        x_unique_values[1:] + x_unique_values[:-1]
                        ) / 2

            for split_value_cur in self.split_values[col]:
                left_split = y[X[col] <= split_value_cur]
                right_split = y[X[col] > split_value_cur]

                left_count_labels = left_split.value_counts()
                p_left = left_count_labels / left_count_labels.sum()
                s_left = self.__node_rule(p_left, left_split)

                right_count_labels = right_split.value_counts()
                p_right = right_count_labels / right_count_labels.sum()
                s_right = self.__node_rule(p_right, right_split)

                weight_left = len(left_split) / len(y)
                weight_right = len(right_split) / len(y)

                s_cur = weight_left * s_left + weight_right * s_right
                if s_cur_min > s_cur:
                    s_cur_min = s_cur
                    col_name = col
                    split_value = split_value_cur

        ig = s_zero - s_cur_min
        return col_name, split_value, ig

    def __node_rule(self, p, split=pd.Series()):
        """
        Calculate the impurity measure (entropy or gini) for a node.

        Args:
            p (pd.Series): The proportion of samples for each class.
            split (pd.Series, optional): The split dataset. Defaults to an empty Series.

        Returns:
            float: The calculated impurity measure.
        """
        if self.criterion == 'entropy':
            return -np.sum(p * np.log2(p)) if not split.empty else 0
        elif self.criterion == 'gini':
            return 1 - np.sum(p ** 2)

    def __str__(self):
        """
        Return a string representation of the MyTreeClf object.

        Returns:
            str: A string containing the class name and its parameters.
        """
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "MyTreeClf class: " + ", ".join(params)

# Description of the classes and their functionality:

# Node class:
# This class represents a node in our decision tree. Each node can be either a decision node
# (with a feature and split value) or a leaf node (with a prediction value). It's the building
# block of our tree structure.

# MyTreeClf class:
# This is our main Decision Tree Classifier class. It's an implementation that allows
# you to create, train, and use a decision tree for classification tasks. Here's what it can do:

# 1. Build a tree: Use the 'fit' method to construct a decision tree from your training data.
# 2. Make predictions: Once trained, you can use 'predict' to classify new data, or 'predict_proba'
#    to get class probabilities.
# 3. Visualize the tree: The 'print_tree' method lets you see the structure of your trained tree.
# 4. Customize your tree: You can adjust various parameters like max depth, minimum samples to split,
#    maximum number of leaves, and even choose between entropy or gini impurity as your split criterion.

# This implementation is great for learning about decision trees or for when you need a customizable,
# transparent classifier. Enjoy exploring it if you're a beginner.
# Otherwise, I'm flattered you've made it this far!
