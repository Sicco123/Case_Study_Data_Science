import pandas as pd
import numpy as np
import pickle
from Junk.time_varying_coefficient_estimation import transform_data
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(2022)

def get_and_prepare_data(column):
    source = 'reproduction_vs_index_Japan.pkl'
    with open(source, 'rb') as f:
        raw = pickle.load(f)

    data = raw.dropna()  # removes nans
    y_log, X_log = transform_data(data['Rt'], data[column])
    y_log, X_log = np.array(y_log), np.array(X_log)
    X = np.vstack((np.ones((1, len(X_log))), X_log.reshape(1, -1)))

    time_steps = np.array(data['time_index'] / len(data))

    return y_log, X, time_steps

def kernel_function(u):
    """
    kernel_function: returns the function value of the Epanechnikov kernel with u as the argument
                  u: float
    """

    if abs(u) <= 1:
        res = (3 / 4) * (1 - u ** 2)
    else:
        res = 0
    return res


def compute_weight(k, time_steps, tau, h):
    t_array = time_steps - tau
    k_i = lambda t: kernel_function(t / h)
    vfunc = np.vectorize(k_i, otypes=[float])
    K = vfunc(t_array)
    weight = (t_array ** k) * (K / h)

    return weight


def compute_S_k(X, k, time_steps, tau, h):
    n = len(time_steps)
    weight = compute_weight(k, time_steps, tau, h)
    W = np.zeros((n, n))
    np.fill_diagonal(W, weight)
    S = (1/n) * np.matmul(X, np.matmul(W, X.T))


    return S


def compute_T(X, Y, k, time_steps, tau, h):
    n = len(time_steps)
    weight = compute_weight(k, time_steps, tau, h)
    T = (1 / n) * np.matmul(X, (weight * Y).reshape(-1, 1))

    return T


def compute_theta(X, Y, time_steps, tau, h):
    S0 = compute_S_k(X, 0, time_steps, tau, h)
    S1 = compute_S_k(X, 1, time_steps, tau, h)
    S2 = compute_S_k(X, 2, time_steps, tau, h)
    T0 = compute_T(X, Y, 0, time_steps, tau, h)
    T1 = compute_T(X, Y, 1, time_steps, tau, h)

    S_top = np.hstack((S0, S1.T))
    S_bottom = np.hstack((S1, S2))
    S = np.vstack((S_top, S_bottom))
    T = np.vstack((T0, T1))

    theta = np.matmul(np.linalg.inv(S), T)

    return theta

def local_linear_estimation(y, X, time_steps, steps, h):
    theta_all = np.empty((len(steps), 2))
    for i, step in enumerate(steps):
        result = compute_theta(X, y, time_steps, step, h)
        theta_all[i] = result[:2].T

    return theta_all

def check_outliers(theta):
    q3, q1 = np.percentile(theta, [75 ,25])
    iqr = q3 - q1
    max = q3+1.5*iqr
    num_outliers = len(theta[theta>max])
    if num_outliers != 0:
        return True
    else:
        return False


def main():
    # column = 'StringencyIndex'   #0.07 = 0.14
    column = 'GovernmentResponseIndex'   #0.06 = 0.24
    # column = 'ContainmentHealthIndex'   #0.04 = 0.13

    y, X, time_steps = get_and_prepare_data(column)
    # steps = np.random.uniform(low=0, high=1, size=(1000,))
    steps = time_steps
    steps.sort()

    h = 0.13

    theta_estimate = local_linear_estimation(y, X, time_steps, steps, h)

    # print(theta_estimate[:-25, 1])
    plt.plot(theta_estimate[:, 1])
    plt.ylim(-60, 60)
    plt.title(f'beta 1 for {h}')
    plt.show()

    not_found = False
    # plt.plot(theta_estimate[:, 0])
    # plt.title('beta 0')
    # plt.show()
    #
    #
    # y_pred = np.dot(X.T, theta_estimate.T)
    # y_pred = np.diagonal(y_pred)
    # plt.title('y pred')
    # plt.plot(y_pred)
    # plt.scatter(time_steps*458, y, s=1, color='red')
    # plt.show()


if __name__ == '__main__':
    main()
