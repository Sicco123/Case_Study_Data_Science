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
from local_linear_estimation_cai import compute_S_k
from local_linear_estimation_cai import compute_weight
import math
import pickle
from tqdm import tqdm

def compute_T_prime_K(X, time_steps, tau, h, k):
    n = len(time_steps)
    weight = compute_weight(k, time_steps, tau, h)
    W = np.zeros((n, n))
    np.fill_diagonal(W, weight)
    T_k = (1/n) * np.matmul(X, W)

    return T_k


def compute_T_prime(X, time_steps, tau, h):
    T0 = compute_T_prime_K(X, time_steps, tau, h, 0)
    T1 = compute_T_prime_K(X, time_steps, tau, h, 1)

    T_prime = np.vstack([T0, T1])

    return T_prime

def compute_S(X, time_steps, tau, h):
    S0 = compute_S_k(X, 0, time_steps, tau, h)
    S1 = compute_S_k(X, 1, time_steps, tau, h)
    S2 = compute_S_k(X, 2, time_steps, tau, h)
    S_top = np.hstack((S0, S1.T))
    S_bottom = np.hstack((S1, S2))
    S = np.vstack((S_top, S_bottom))
    S = np.linalg.inv(S)

    return S

def compute_H_row(x, X, identity, time_steps, tau, h):
    S = compute_S(X, time_steps, tau, h)
    T_prime = compute_T_prime(X, time_steps, tau, h)
    # H_row = np.matmul(np.matmul(x, identity), np.matmul(S, T_prime))
    H_row = np.matmul(x, np.matmul(S, T_prime))


    return H_row

def compute_n_h(X, time_steps, h):
    p, n = X.shape
    X_tilde = np.hstack([X.T, np.zeros((n, p))])  #TODO: maybe replace with Z
    identity = np.zeros((2*p, 2*p))
    identity_diagonal = np.array([1]*p + [0]*p)
    np.fill_diagonal(identity, identity_diagonal)

    H_matrix = []
    for i, tau in enumerate(time_steps):
        H_row = compute_H_row(X_tilde[i].reshape(1,-1), X, identity, time_steps, tau, h)
        H_matrix.append(H_row)

    H_matrix = np.stack(H_matrix).reshape(n, n)
    n_h = np.trace(H_matrix)

    return n_h, H_matrix


def compute_aic(y, X, y_pred, time_steps, h):
    n = len(y)

    n_h, H_matrix = compute_n_h(X, time_steps, h)

    # middle = np.matmul((np.identity(n) - H_matrix).T, (np.identity(n) - H_matrix))
    # RSS = np.matmul(y.reshape(-1, 1).T, np.matmul(middle, y.reshape(-1, 1))) * 1/n
    #
    # aic = math.log(RSS) + (2 * (1 + n_h) / (n - n_h - 2))

    sigma_sq = ((y - y_pred) ** 2).mean()
    print(f'trace = {n_h}')
    print(f'sigma sq = {sigma_sq}')
    print('Penalty =', (2 * (n_h + 1) / (n - n_h - 2)))
    aic = math.log(sigma_sq) + (2 * (n_h + 1) / (n - n_h - 2))
    # aic = math.log(sigma_sq)

    return aic


def main():
    bandwidth_values = np.linspace(0.01, 1, 100)
    y, X, time_steps = get_and_prepare_data()
    # steps = np.random.uniform(low=0, high=1, size=(1000,)
    steps = time_steps
    steps.sort()
    results = []

    for bw in tqdm(bandwidth_values):
        theta_estimate = local_linear_estimation(y, X, time_steps, steps, bw)
        y_pred = np.diagonal(np.matmul(X.T, theta_estimate.T))
        aic = compute_aic(y, X, y_pred, time_steps, bw)
        results.append((bw, aic))
        print(f'AIC for bw of {bw} = {aic}')
    #
    with open('bw_selection_2007.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
