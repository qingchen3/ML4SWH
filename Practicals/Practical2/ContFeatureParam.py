from scipy.stats import norm


class ContFeatureParam:

    def __init__(self):
        self.mu = None
        self.sigma = None

    def estimate(self, X):
        self.mu, self.sigma = norm.fit(X)  # loc is the mean and scale is the standard deviation.
        if self.sigma == 0.0:
            self.sigma = 0.0000001

    def get_log_probability(self, X_new):
        return norm(self.mu, self.sigma).logpdf(X_new)
