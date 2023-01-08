from scipy.stats import bernoulli
import numpy as np


class BinFeatureParam:
    bin_model = None
    p = None

    def estimate(self, X):
        self.p = np.sum(X) / len(X)
        self.bin_model = bernoulli(self.p)

    def get_log_probability(self, X_new):
        # probs = []
        # for x_new in X_new:
        #    probs.append(self.bin_model.logpmf(x_new))
        return self.bin_model.logpmf(X_new)
