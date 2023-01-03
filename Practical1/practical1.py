import numpy as np
import matplotlib.pyplot as plt
import _pickle as cp
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score



# the function takes the y-values in the training data as the input and makes the bar chart.
def plot_bar_chart_score(y):
    fix, ax = plt.subplots()
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel('Score')
    plt.ylabel('Number of wines')
    plt.title('Distribution of scores of wines')
    plt.show()


def split_data(X, y, split_coeff=0.8):
    split = int(X.shape[0] * split_coeff)
    X_train = X[: split]
    y_train = y[: split]
    X_test = X[split:]
    y_test = y[split:]
    return X_train, y_train, X_test, y_test


def compute_average(y_train):
    return np.mean(y_train)


# The trivial predictor returns the average value.
def trivial_predictor(X_test, y_train_avg):
    return y_train_avg


def test_predictor(X, y, predictor: callable = None):
    # Apply the predictor to each row of the matrix X to get the predictions
    y_predicted = np.apply_along_axis(predictor, 1, X)
    #for i1, i2 in zip(y, y_predicted):
    #    print(i1, i2)
    mse = np.mean((y - y_predicted) ** 2)
    return mse


def standardize_data(X):

    mean = X.mean(axis = 0)
    std = np.std(X, axis = 0)
    X_std = (X - mean) / std
    return X_std, mean, std


def expand_with_ones(X):
    X_out = np.hstack((np.ones((X.shape[0], 1)), X))
    return X_out


def least_squares_compute_parameters(X_input, y):
    # add the bias column to the data
    X = expand_with_ones(X_input)
    reverse = np.linalg.inv(X.T.dot(X))
    w = reverse.dot(X.T)
    w = w.dot(y)
    return w


def linear_model_predictor(X, w):
    return X.dot(w)


def train_and_test(X_train, y_train, X_test, y_test):
    X_train_std, _, _ = standardize_data(X_train)
    y_train_std, _, _ = standardize_data(y_train)
    w = least_squares_compute_parameters(X_train_std, y_train_std)
    mse_train = test_predictor(expand_with_ones(X_train_std), y_train_std, lambda x: linear_model_predictor(x, w))

    X_test_std, _, _ = standardize_data(X_test)
    y_test_std, _, _ = standardize_data(y_test)
    mse_test = test_predictor(expand_with_ones(X_test_std), y_test_std, lambda x: linear_model_predictor(x, w))
    return mse_train, mse_test


def expand_basis(X, degree):
    poly = PolynomialFeatures(degree)
    ex = poly.fit_transform(X)
    return ex


def prepare_data(X, y, degree):
    # Hints: follow the steps
    # 1. split the data (X, y) into training data (X_train, y_train) and test data (X_test, y_test)
    # 2. standardize the training data and do the same transformation to the test data
    # 3. expand the basis of the training data and test data
    # 4. split the expanded training data into training data (X_train_n, y_train_n) and validation data (X_train_v, y_train_v)

    X_train, y_train, X_test, y_test = split_data(X, y, 0.8)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_train_expanded = expand_basis(X_train_scaled, degree)
    X_train_n, y_train_n, X_train_v, y_train_v = split_data(X_train_expanded, y_train, 0.8)

    scaler = StandardScaler()
    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)
    X_test_expanded = expand_basis(X_test_scaled, degree)

    return X_train_expanded, y_train, X_train_n, y_train_n, X_train_v, y_train_v, X_test_expanded, y_test


def choose_hyper_param(X_train_n, y_train_n, X_train_v, y_train_v, is_ridge: bool):
    mse_arr = []
    lam_arr = []

    # Try lambda values from 10^-4 to 10^2.
    # Record the mse and the lambda values in mse_arr and lam_arr

    for pow_lam in range(-4, 3):
        lam = 10 ** pow_lam
        if is_ridge:
            reg = Ridge(alpha=lam)
        else:
            reg = Lasso(alpha=lam)
        reg.fit(X_train_n, y_train_n)
        predicted = reg.predict(X_train_v)
        mse = calculate_mse(y_train_v, predicted)
        mse_arr.append(mse)
        lam_arr.append(lam)

    # get the index of the lambda value that has the minimal use
    lambda_idx_min = np.argmin(np.array(mse_arr))

    # plot of the lambda values and their mse
    # plt.figure()
    # plt.semilogx(lam_arr, mse_arr)
    # plt.show()

    return lam_arr[lambda_idx_min]


def Train_KFold(X, y):
    minimum_mse = 100000000
    parameters = {}
    for degree in range(1, 5):

        k = 5
        kf = KFold(n_splits=k, random_state=None)

        for pow_lam in range(-4, 3):
            mse_arr = []
            lam = 10 ** pow_lam
            # if is_ridge:
            #    model = Ridge(alpha=lam)
            #else:
            #    model = Lasso(alpha=lam)
            model = Ridge(alpha=lam)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_train_expanded = expand_basis(X_train_scaled, degree)

                scaler = StandardScaler()
                scaler.fit(X_test)
                X_test_scaled = scaler.transform(X_test)
                X_test_expanded = expand_basis(X_test_scaled, degree)

                model.fit(X_train_expanded, y_train)
                pred_values = model.predict(X_test_expanded)

                mse = calculate_mse(y_test, pred_values)
                mse_arr.append(mse)
                if mse < minimum_mse:
                    minimum_mse = mse
                    parameters['lam'] = lam
                    parameters['degree'] = degree
                # print("degree: %d, lamba:%d, mse:%f" % (degree, lam, mse))
            # get the index of the lambda value that has the minimal use
            lambda_idx_min = np.argmin(np.array(mse_arr))
        # print()
        # print("degree: %d, lamba:%d, minimum mse:%f" %(degree, lam, mse_arr[lambda_idx_min]))
        # print()
    print("\n\nTrain and test with Kfold:")
    print("optimal lambada:%d, optimal degree: %d, minimum mse:%f" %(parameters['lam'], parameters['degree'], minimum_mse))


def calculate_mse(y, y_predicted):
    return np.mean((y - y_predicted) ** 2)


if __name__ == '__main__':
    X, y = cp.load(open('winequality-white.pickle', 'rb'))

    # check the size of the data
    print("X is a matrix with the dimension {}. That is, {} data records and {} features.".format(X.shape, X.shape[0],
                                                                                                  X.shape[1]))
    print("y is a vector with {} values. They are the labels of the data records in X.".format(y.shape[0]))

    X_train, y_train, X_test, y_test = split_data(X, y, 0.8)  # split the data with split_coeff=0.8


    y_train_avg = compute_average(y_train)
    print("The average of the y-values in the training data is {}".format(y_train_avg))

    # use the function test_predictor to test the trivial predictor
    # we use the lambda function here to pass the function trivial predictor to the function test_predictor.
    mse_trivial_predictor_train = test_predictor(X_train, y_train, lambda x: trivial_predictor(x, y_train_avg))
    mse_trivial_predictor_test = test_predictor(X_test, y_test, lambda x: trivial_predictor(x, y_train_avg))

    # Report the result
    print('Trivial Predictor')
    print('MSE (Training) = %.4f' % mse_trivial_predictor_train)
    print('MSE (Testing)  = %.4f' % mse_trivial_predictor_test)
    print('--------------------------------------------------------------------------------\n')

    # X_train_std, X_train_mean, X_train_std_div = standardize_data(X_train)

    # print("Mean:", X_train_mean)
    # print("Standard deviation:", X_train_std_div)

    # w = least_squares_compute_parameters(X_train_std, y_train)
    # print("w:", w.shape)

    # X_test_std, X_test_mean, X_test_std_div = standardize_data(X_test)
    # mse_linear_model_predictor = test_predictor(expand_with_ones(X_test_std), y_test,
    #                                            lambda x: linear_model_predictor(x, w))

    mse_train, mse_test = train_and_test(X_train, y_train, X_test, y_test)
    print("Mean squared error is {} in training".format(mse_train))
    print("Mean squared error is {} in testing".format(mse_test))

    '''
    mse_train_v = []
    mse_test_v = []

    TRAINING_SIZE_MAX = 601
    TRAINING_SIZE_MIN = 20

    # compute the MSE over data with sizes from TRAINING_SIZE_MIN to TRAINING_SIZE_MAX with increasing step 20
    for train_size in range(TRAINING_SIZE_MIN, TRAINING_SIZE_MAX, 20):

        mse_train, mse_test = train_and_test(X_train[:train_size], y_train[:train_size], X_test, y_test)

        mse_train_v.append(mse_train)
        mse_test_v.append(mse_test)

    plt.plot(np.arange(TRAINING_SIZE_MIN, TRAINING_SIZE_MAX, 20), mse_train_v, 'r--', label="Training Error")
    plt.plot(np.arange(TRAINING_SIZE_MIN, TRAINING_SIZE_MAX, 20), mse_test_v, 'b-', label="Test Error")
    plt.legend(loc="upper right")
    plt.xlabel('Dataset Size')
    plt.ylabel('Mean Squared Error')
    plt.show()
    '''

    X_train, y_train, X_train_n, y_train_n, X_train_v, y_train_v, X_test, y_test = prepare_data(X, y, 2)

    # check the size of the splitted dataset
    # print("Shape of X_train_n:", X_train_n.shape)  # expected output (3134, 78)
    # print("Shape of y_train_n:", y_train_n.shape)  # expected output (3134,)
    # print("Shape of X_train_v:", X_train_v.shape)  # expected output (784, 78)
    # print("Shape of y_train_v:", y_train_v.shape)  # expected output (784,)
    # print("Shape of X_test:", X_test.shape)  # expected output (980, 78)
    # print("Shape of y_test:", y_test.shape)  # expected output (980,)

    # call the function to choose the lambda for Ridge and Lasso
    lam_ridge = choose_hyper_param(X_train_n, y_train_n, X_train_v, y_train_v, True)
    lam_lasso = choose_hyper_param(X_train_n, y_train_n, X_train_v, y_train_v, False)

    print("Ridge lambda:", lam_ridge)
    print("Lasso lambda:", lam_lasso)

    Ridge_reg = Ridge(alpha=lam_ridge)
    Ridge_reg.fit(X_train, y_train)

    Ridge_train_predicted = Ridge_reg.predict(X_train)
    Ridge_test_predicted = Ridge_reg.predict(X_test)
    mse_ridge_train = calculate_mse(y_train, Ridge_train_predicted)
    mse_ridge_test = calculate_mse(y_test, Ridge_test_predicted)

    Lasso_reg = Lasso(alpha=lam_lasso)
    Lasso_reg.fit(X_train, y_train)

    Lasso_train_predicted = Lasso_reg.predict(X_train)
    Lasso_test_predicted = Lasso_reg.predict(X_test)

    mse_lasso_train = calculate_mse(y_train, Lasso_train_predicted)
    mse_lasso_test = calculate_mse(y_test, Lasso_test_predicted)

    # Report the result
    print('\n\nFor Ridge Regression with using degree %d polynomial expansion and lambda = %.4f' % (2, lam_ridge))
    print('--------------------------------------------------------------------------------')
    print('MSE (Training) = %.4f' % mse_ridge_train)
    print('MSE (Testing)  = %.4f' % mse_ridge_test)

    print('\n\nFor Lasso with using degree %d polynomial expansion and lambda = %.4f' % (2, lam_lasso))
    print('---------------------------------------------------------------------')
    print('MSE (Training) = %.4f' % mse_lasso_train)
    print('MSE (Testing)  = %.4f' % mse_lasso_test)

    Train_KFold(X, y)




