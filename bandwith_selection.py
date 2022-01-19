"""
Authors: Jamie Rutgers, Mirte Pruppers, Damiaan Bonnet, Sicco Kooiker
Name: static_coefficient_estimation.py
Date: 12-1-2022
Desription: In this file we present various methods to derive the optimal bandwith.
These bandwiths we use in "time_varying_coefficient_estimation.py"
Calculations are based on https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.6363&rep=rep1&type=pdf
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from local_linear_estimation_cai import get_and_prepare_data
from local_linear_estimation_cai import local_linear_estimation
from local_linear_estimation_cai import kernel_function
import math


def compute_W(time_steps, t, h):
    t_array = time_steps - t
    w_i = lambda t: kernel_function(t / h)
    vfunc = np.vectorize(w_i, otypes=[float])
    diagonal = vfunc(t_array)
    diagonal = diagonal / h

    W = np.zeros((len(time_steps), len(time_steps)))
    np.fill_diagonal(W, diagonal)

    return W


def compute_X_tilde(X, time_steps, t):
    t_array = time_steps - t
    X_time = X * t_array.reshape(-1, 1)
    X_tilde = np.hstack([X, X_time])

    return X_tilde


def compute_A(X, time_steps, t, h):
    p = X.shape[1]
    I = np.identity(p)
    zero_matrix = np.zeros((p, p))
    I_weight = np.hstack([I, zero_matrix])
    X_tilde = compute_X_tilde(X, time_steps, t)
    W = compute_W(time_steps, t, h)
    A1 = np.linalg.inv(np.matmul(X_tilde.T, np.matmul(W, X_tilde)))
    A2 = np.matmul(X_tilde.T, W)

    A = np.matmul(I_weight, np.matmul(A1, A2))

    return A


def compute_n_h(X, time_steps, h):
    S_star = []
    for i, t in enumerate(time_steps):
        A = compute_A(X, time_steps, t, h)
        S_star.append(np.matmul(A.T, X[i]))

    S_star = np.stack(S_star).T
    n_h = np.trace(S_star)

    return n_h


def compute_aic(y, X, theta_estimate, time_steps, h):
    n = len(y)
    y_pred = np.dot(X.T, theta_estimate.T)
    y_pred = np.diagonal(y_pred)
    sigma_sq = ((y - y_pred) ** 2).mean()
    n_h = compute_n_h(X.T, time_steps, h)

    aic = math.log(sigma_sq) + (2 * (n_h + 1) / (n - n_h - 2))

    return aic


def main():
    bandwidth_values = [0.15, 0.2, 0.1, 0.05]
    y, X, time_steps = get_and_prepare_data()
    steps = np.random.uniform(low=0, high=1, size=(1000,))
    steps.sort()

    for bw in bandwidth_values:
        theta_estimate = local_linear_estimation(y, X, time_steps, steps, bw)
        aic = compute_aic(y, X, theta_estimate, time_steps, bw)

        print(f'AIC for bw of {bw} = {aic}')


if __name__ == '__main__':
    main()
