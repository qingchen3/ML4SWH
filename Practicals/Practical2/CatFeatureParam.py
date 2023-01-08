from scipy.stats import multinomial
import numpy as np


class CatFeatureParam:

    def __init__(self, num_of_categories):
        self.num_of_categories = num_of_categories
        self.Cat_model = None

    def estimate(self, X):
        count_arr = np.bincount(X)
        p_arr = count_arr / len(X)
        self.Cat_model = multinomial(1, p_arr)

    def get_log_probability(self, X_new):
        return self.Cat_model.logpmf(X_new)