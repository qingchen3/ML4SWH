# import pylab
# pylab.rcParams['figure.figsize'] = (10., 10.)

import pickle as cp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from NBC import NBC
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from sklearn.datasets import load_iris


ALPHA = 1.0 # for additive smoothing


def compareNBCvsLR(nbc, lr, X, y, num_runs=200, num_splits=10):
    tst_errs_nbc = np.zeros(num_splits)
    tst_errs_lr = np.zeros(num_splits)

    if 'r' in nbc.feature_types:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)

    for i in range(num_runs):
        N, D = X.shape

        Ntrain = int(0.8 * N)
        shuffler = np.random.permutation(N)
        Xtrain = X[shuffler[:Ntrain]]
        ytrain = y[shuffler[:Ntrain]]
        Xtest = X[shuffler[Ntrain:]]
        ytest = y[shuffler[Ntrain:]]
        for split in range(num_splits):
            Ntrain_size = int(((split + 1)/num_splits) * Ntrain)
            Xtrain_sample = Xtrain[:Ntrain_size, :]
            ytrain_sample = ytrain[:Ntrain_size]

            nbc.fit(Xtrain_sample, ytrain_sample)
            yhat = nbc.predict(Xtest)
            tst_errs_nbc[split] += (1 - np.mean(yhat == ytest))

            lr.fit(Xtrain_sample, ytrain_sample)
            yhat = lr.predict(Xtest)
            tst_errs_lr[split] += (1 - np.mean(yhat == ytest))

    return tst_errs_nbc / num_runs, tst_errs_lr/num_runs


def makePlot(tst_errs_nbc, tst_errs_lr, title=None, num_splits=10):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params(axis='both', labelsize=20)

    ax.set_xlabel('Percent of training data used', fontsize=20)
    ax.set_ylabel('Classification Error', fontsize=20)
    if title is not None: ax.set_title(title, fontsize=25)

    xaxis_scale = [(i + 1) * (100 / num_splits) for i in range(num_splits)]
    plt.plot(xaxis_scale, tst_errs_nbc, label='Naive Bayes')
    plt.plot(xaxis_scale, tst_errs_lr, label='Logistic Regression', linestyle='dashed')

    ax.legend(loc='upper right', fontsize=20)
    plt.show()


if __name__ == '__main__':
    # X = np.array([2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2.,  1.9, 2.1, 2.,  2.4, 2.3, 1.8, 2.2, 2.3, 1.5])

    # param = ContFeatureParam()
    # param.estimate(X)
    # print("mu and sigma: ", param.mu, param.sigma)
    # probs = param.get_log_probability(np.array([0, 1, 2, 3]))
    # print(probs)

    # X = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    # param = BinFeatureParam()
    # param.estimate(X)
    # probs = param.get_log_probability(np.array([0, 1]))
    # print(probs)

    # X = np.array([0, 6, 5, 4, 0, 6, 6, 4, 1, 1, 2, 3, 8, 8, 1, 6, 4, 9, 0, 2, 2, 3, 8, 0, 2])
    # param = CatFeatureParam(num_of_categories=10)
    # param.estimate(X)
    # print(param.get_log_probability([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    print("Testing NBC for continous values")
    iris = load_iris()
    X, y = iris['data'], iris['target']

    N, D = X.shape
    Ntrain = int(0.8 * N)
    Xtrain = X[:Ntrain]
    ytrain = y[:Ntrain]
    Xtest = X[Ntrain:]
    ytest = y[Ntrain:]

    nbc_iris = NBC(feature_types=['r', 'r', 'r', 'r'])
    nbc_iris.fit(Xtrain, ytrain)
    yhat = nbc_iris.predict(Xtest)
    test_accuracy = np.mean(yhat == ytest)

    print("Accuracy:%f\n\n" % test_accuracy)  # should be larger than 90%

    # ----------------------------------------------------------------------
    print("Testing NBC for Binary values")
    data = pd.read_csv('binary_test.csv', header=None)
    data = data.to_numpy()

    X = data[:, 1:]
    y = data[:, 0]

    N, D = X.shape
    Ntrain = int(0.8 * N)
    Xtrain = X[:Ntrain]
    ytrain = y[:Ntrain]
    Xtest = X[Ntrain:]
    ytest = y[Ntrain:]

    nbc = NBC(feature_types=['b'] * 16)
    nbc.fit(Xtrain, ytrain)
    yhat = nbc.predict(Xtest)
    test_accuracy = np.mean(yhat == ytest)

    print("Accuracy:%f\n\n" % test_accuracy) # should be larger than 85%

    # ----------------------------------------------------------------------
    print("Compare NBC and lr on iris dataset")
    iris = load_iris()
    X, y = iris['data'], iris['target']
    nbc = NBC(feature_types=['r', 'r', 'r', 'r'])
    lr = LogisticRegression(random_state=0)

    tst_errs_nbc, tst_errs_lr = compareNBCvsLR(nbc, lr, X, y, num_runs=200, num_splits=10)
    makePlot(tst_errs_nbc, tst_errs_lr, title=None, num_splits=10)

    # ----------------------------------------------------------------------

    print("\nCompare NBC and lr on voting dataset")
    voting = pd.read_csv('voting.csv')
    voting = voting.dropna()
    voting.replace(('y', 'n'), (1, 0), inplace=True)
    voting.replace(('republican', 'democrat'), (1, 0), inplace=True)
    voting = voting.to_numpy()
    y = voting[:, 0]
    X = voting[:, 1:]
    print(X.shape, y.shape)

    nbc = NBC(feature_types=['b'] * 16)
    lr = LogisticRegression(random_state=0)

    tst_errs_nbc, tst_errs_lr = compareNBCvsLR(nbc, lr, X, y, num_runs=200, num_splits=10)
    makePlot(tst_errs_nbc, tst_errs_lr, title=None, num_splits=10)