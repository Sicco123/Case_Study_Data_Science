import pandas as pd
import numpy as np
import pickle
from time_varying_coefficient_estimation import transform_data
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(2022)

def kernel_function(u):
    """
    kernel_function: returns the function value of the Epanechnikov kernel with u as the argument
                  u: float
    """

    if abs(u) < 1:
        res = (3 / 4) *(1 - u ** 2)
    else:
        res = 0
    return res


def compute_S(X, k, time_steps, tau, h):
    n = len(time_steps)
    sum = 0
    for i in range(n):
        weight = (time_steps[i]**k)*(kernel_function((time_steps[i]-tau)/h)/h)
        sum += np.matmul(X[:,i].reshape(-1, 1), X[:,i].reshape(-1, 1).T)*weight

    S = sum/n

    return S

def compute_T(X, Y, k, time_steps, tau, h):
    n = len(time_steps)
    sum = 0
    for i in range(n):
        weight = (time_steps[i]**k)*(kernel_function((time_steps[i]-tau)/h)/h)
        sum += X[:,i].reshape(-1,1)*weight*Y[i]

    T = sum/n

    return T


def compute_theta(X, Y, time_steps, tau, h):
    X = np.vstack((np.ones((1, len(X))), X.reshape(1, -1)))
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

    if theta[1] > 100:
        print('debug')

    return theta

def main():
    source = 'reproduction_vs_index_Japan.pkl'
    with open(source, 'rb') as f:
        raw = pickle.load(f)

    data = raw.dropna()  # removes nans
    data = data.iloc[0:458] # bad slicing, works for now
    y_log, X_log = transform_data(data['Rt'], data['StringencyIndex'])
    y_log, X_log = np.array(y_log), np.array(X_log)
    steps = np.random.uniform(low=0, high=1, size=(1000,))
    steps.sort()
    time_steps = np.array(data['time_index'] / len(data))
    h = 0.05
    theta_all = np.empty((len(steps), 2))
    for i, step in enumerate(tqdm(steps)):
        result = compute_theta(X_log, y_log, time_steps, step, h)
        theta_all[i] = result[:2].T

    print(theta_all[:,1])

    plt.plot(theta_all[:,1])
    plt.show()
if __name__ == '__main__':
    main()