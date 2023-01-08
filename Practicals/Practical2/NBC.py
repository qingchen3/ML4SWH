from ContFeatureParam import ContFeatureParam
from BinFeatureParam import BinFeatureParam
from CatFeatureParam import CatFeatureParam
import numpy as np


class NBC:
    # Inputs:
    # feature_types: the array of the types of the features, e.g., feature_types=['b', 'r', 'c']
    def __init__(self, feature_types):
        self.feature_types = feature_types
        self.theta = []
        self.pi = None

    # The function uses the input data to estimate all the parameters of the NBC
    def fit(self, X, y):
        C, counts = np.unique(y, return_counts=True)
        y_size = len(y)
        self.pi = np.zeros(max(y) + 1)
        for c, c_count in zip(C, counts):
            self.pi[c] = c_count / y_size
            self.theta.append([])
            for j in range(len(self.feature_types)):
                if 'r' in self.feature_types:
                    self.theta[c].append(ContFeatureParam())
                if 'b' in self.feature_types:
                    self.theta[c].append(BinFeatureParam())
                self.theta[c][j].estimate(X[y == c, j])

    # The function takes the data X as input, and predicts the class for the data
    def predict(self, X):
        res = None
        for c in range(len(self.pi)):
            row_res = np.zeros(X.shape[0])
            for j in range(X.shape[1]):
                row_res += self.theta[c][j].get_log_probability(X[:, j])
            row_res += self.pi[c]
            if res is None:
                res = row_res
            else:
                res = np.vstack((res, row_res))
        return np.argmax(res, axis = 0)