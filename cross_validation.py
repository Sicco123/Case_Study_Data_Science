import pandas as pd
import numpy as np
import pickle
from Junk.time_varying_coefficient_estimation import transform_data
import matplotlib.pyplot as plt


np.random.seed(2022)

def get_and_prepare_data(source, index):
    with open(source, 'rb') as f:
        raw = pickle.load(f)

    data = raw.dropna()  # removes nans
    y_log, X_log = transform_data(data['Rt'], data[data.columns[index]])
    y_log, X_log = np.array(y_log), np.array(X_log)
    X = np.vstack((np.ones((1, len(X_log))), X_log.reshape(1, -1)))

    time_steps = np.array(data['time_index'] / len(data))

    return y_log, X, time_steps, data.columns[index]

def kernel_function(u):
    """
    kernel_function: returns the function value of the Epanechnikov kernel with u as the argument
                  u: float
    """

    if abs(u) < 1:
        res = (3 / 4) * (1 - u ** 2)
    else:
        res = 0
    return res


def compute_weight(k, time_steps, tau, h):
    t_array = time_steps - tau
    t_array[np.where(time_steps == tau)] = 0 # leave one out
    k_i = lambda t: kernel_function(t / h)
    vfunc = np.vectorize(k_i, otypes=[float])
    K = vfunc(t_array)
    weight = (t_array ** k) * (K / h)

    return weight


def compute_S(X, k, time_steps, tau, h):
    n = len(time_steps)
    weight = compute_weight(k, time_steps, tau, h)
    S = (1 / n) * np.matmul(X, X.T * weight.reshape(-1, 1))

    return S


def compute_T(X, Y, k, time_steps, tau, h):
    n = len(time_steps)
    weight = compute_weight(k, time_steps, tau, h)
    T = (1 / n) * np.matmul(X, (weight * Y).reshape(-1, 1))

    return T


def compute_theta(X, Y, time_steps, tau, h):
    S0 = compute_S(X, 0, time_steps, tau, h)
    S1 = compute_S(X, 1, time_steps, tau, h)
    S2 = compute_S(X, 2, time_steps, tau, h)
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

def cross_validation_bandwith(u1, u2, y, X, steps, time_steps):
    h_array = np.linspace(u1,u2,100) # 100 steps

    log_lik_opt = 100000
    results = []

    for h in h_array:
        theta_estimate = local_linear_estimation(y, X, time_steps, steps, h)
        # plt.plot(theta_estimate[:, 1])
        # plt.title('beta 1')
        # plt.show()

        y_pred = np.diagonal(X.T @ theta_estimate.T)

        resid = y - y_pred

        log_likelihood = np.log(np.sum(np.square(resid)))

        results.append((h,log_likelihood))
        if log_likelihood < log_lik_opt:
            log_lik_opt = log_likelihood
            h_opt = h

        print(f'The log likelihood of bandwith ({h}) = {log_likelihood}')

    return h_opt, results

def main():
    source = 'reproduction_vs_index_Japan.pkl'
    index = 3#'StringencyIndex'
    y, X, time_steps, index_name = get_and_prepare_data(source, index)
    steps = time_steps
    steps.sort()

    h_opt, results = cross_validation_bandwith(0.01, 0.2, y, X, steps, time_steps)

    with open(f'cv_bw_selection_{index_name}.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
