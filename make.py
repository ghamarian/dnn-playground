import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


X, Y = make_classification(n_samples=50000, n_features=10, n_informative=8,
                           n_redundant=0, n_clusters_per_class=2)
Y = np.array([Y, -(Y-1)]).T  # The model currently needs one column for each class
X, X_test, Y, Y_test = train_test_split(X, Y)

print(X)