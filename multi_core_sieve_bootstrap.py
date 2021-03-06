"""
Authors: Jamie Rutgers, Mirte Pruppers, Damiaan Bonnet, Sicco Kooiker
Name: static_coefficient_estimation.py
Date: 12-1-2022
Desription: In this file we give a wild bootstrap procedure to obtain confidence intervals
for the time varying coefficient estimates.
"""
import pandas as pd
import numpy as np
import pickle
from Junk.time_varying_coefficient_estimation import transform_data
import matplotlib.pyplot as plt
from tqdm import tqdm
import local_linear_estimation_cai as ll_estimation
import cross_validation as cv
import multiprocessing
import math
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from statsmodels.tsa.ar_model import AutoReg, ar_select_order


def two_point(B, n):
    normal_draws =  np.random.normal(0,1,size = (B, n))
    normal_draws[np.where(normal_draws < 0)] = -1
    normal_draws[np.where(normal_draws >= 0)] = 1
    two_point_draws = normal_draws
    return two_point_draws

def simulate_residuals(residuals, B, max_lag):

    mod = ar_select_order(residuals, maxlag=max_lag, trend = 'n', glob = True)
    res = mod.model.fit()
    lags = res.ar_lags
    lags = [lag - 1 for lag in lags]
    params = res.params

    innovations = res.resid
    innovations = innovations[(-len(residuals)+len(innovations)+10):]
    innovations = innovations -np.mean(innovations)
    innovations_bootstrap = np.random.choice(innovations, size = (B, len(innovations)), replace = True)
    #innovations_bootstrap = innovations_bootstrap - innovations_bootstrap.mean(axis=1, keepdims=True)

    bootstrap_errors = innovations_bootstrap.copy()

    X_p = np.tile(residuals[:max_lag],(B,1)).T

    for t in range(len(innovations_bootstrap[0,:])):
        bootstrap_errors[:,t] = bootstrap_errors[:,t] + params @ X_p[lags,:]
        X_p[1:max_lag,:] = X_p[0:(max_lag-1),:]
        X_p[0,:] = bootstrap_errors[:,t]

    print(bootstrap_errors.shape)

    return bootstrap_errors

def simulate_y(y_pred, residuals, B, max_lag):

    #normal_draws = np.random.normal(0,1,size = (B, len(y_pred)))

    repeat_residuals = np.repeat(residuals.reshape(-1,1), B, axis = 1 ).T
    repeat_y_pred = np.repeat(y_pred.reshape(-1,1), B, axis = 1 ).T

    resampled_residuals = simulate_residuals(residuals, B, max_lag)#normal_draws * repeat_residuals
    #resampled_residuals -= np.mean(resampled_residuals, axis=0)

    simulated_y_pred = repeat_y_pred[:,max_lag:] + resampled_residuals

    y_mean =simulated_y_pred.mean(axis = 0 , keepdims = True)
    #print(y_mean)
    #plt.plot(y_mean[0])
    #plt.show()
    return simulated_y_pred

def local_linear_estimation(arguments):
    y, X, time_steps, steps, h = list(list(arguments))
    theta_all = np.empty((len(steps), 2))
    for i, step in enumerate(steps):
        result = ll_estimation.compute_theta(X, y, time_steps, step, h)
        theta_all[i] = result[:2].T

    return theta_all


def bootstrap_procedure(y_pred, residuals, beta_estimate, B, steps, time_steps, y, X, bandwidth, max_lag):
    y_simulated = simulate_y(y_pred, residuals, B, max_lag)

    beta_estimate = beta_estimate[max_lag:]
    steps = steps[max_lag:]
    time_steps = time_steps[max_lag:]
    X = X[:,max_lag:]

    beta_bootstrap = np.zeros(((B + 1), len(beta_estimate)))
    beta_bootstrap[0, :] = beta_estimate


    X_repeat = [X] * B
    time_steps_repeat = [time_steps] * B
    steps_repeat = [steps] * B
    bandwidth_repeat = [bandwidth] * B
    input = zip(y_simulated, X_repeat, time_steps_repeat, steps_repeat, bandwidth_repeat)

    a_pool = multiprocessing.Pool(processes=8)
    result = a_pool.map(local_linear_estimation, input)

    for b, res in enumerate(result):
        beta_bootstrap[b + 1, :] = res[:, 1]

    return beta_bootstrap

def plot_beta_CI(x, beta_observed, q_bootstrap, alpha, B, index):

    a1 = 27/448
    b1 = 74/448

    a2 = 303/448
    b2 = 385/448

    a3 = 418/448
    b3 = 448/448

    quantile_1 = int(0.5*alpha*(B+1))
    quantile_2 = int((1-0.5*alpha)*(B+1))

    print(quantile_1)
    print(quantile_2)

    q_bootstrap.sort(axis = 0)
    q_bootstrap_q1 = q_bootstrap[quantile_1,:]
    q_bootstrap_q2 = q_bootstrap[quantile_2,:]

    beta_bootstrap_q1 = beta_observed + q_bootstrap_q1
    beta_bootstrap_q2 = beta_observed + q_bootstrap_q2

    fig, ax = plt.subplots()
    ax.plot(x, beta_observed, color = 'k')
    ax.plot(x, beta_bootstrap_q1, '--', color = 'r', alpha = 0.8)
    ax.plot(x, beta_bootstrap_q2, '--', color = 'r', alpha = 0.8)
    ax.set_ylim(-9,9)
    ax.set_xlim(0,1)
    ax.fill_between(x, beta_bootstrap_q1, beta_bootstrap_q2, color='r', alpha=.05)
    ax.set_xticks([21/448, 51/448, 82/448, 112/448, 143/448, 174/448, 204/448, 235/448, 265/448, 296/448, 327/448, 355/448, 386/448, 416/448, 447/448])
    ax.set_xticklabels([4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6])
    ax.axhline(y=0, color = 'k', linewidth = 0.5)
    #ax.set_title('Time Varying Estimates', fontweight='bold', fontsize=18)
    plt.axvspan(a1, b1, color='k', alpha=0.1, lw=0)
    plt.axvspan(a2, b2, color='k', alpha=0.1, lw=0)
    plt.axvspan(a3, b3, color='k', alpha=0.1, lw=0)
    plt.ylabel("beta_1(t)")
    plt.xlabel("t (month)")
    fig.savefig(f'Figures/Bootstraped_Sieve_Confidence_Intervals_Time_Varying_Estimates_{index}_cut_off')
    plt.show()

def over_smooth_bandwidth(n, bandwidth_2):
    crit_1 = h_tilde
    crit_2 = n * h * h_tilde ** 4
    crit_3 = h * math.log(n / h_tilde)

    return max(crit_1, crit_2, crit_3)

def main():
    bandwidth_2 = 0.13
    bandwidth_1 = 2*bandwidth_2**(5/9)

    #steps_size = 1000
    B = 999
    alpha = 0.05
    max_lag = 10

    source = 'reproduction_vs_index_Japan.pkl'
    index = 3  # 'StringencyIndex'

    y, X, time_steps, name = cv.get_and_prepare_data(source, index)
    steps = time_steps
    steps.sort()

    theta_estimate = local_linear_estimation([y, X, time_steps, steps, bandwidth_1])
    y_pred = np.diagonal(X.T @ theta_estimate.T)
    residuals = y - y_pred
    beta_estimate = theta_estimate[:, 1]

    #plt.plot(y[max_lag:])


    beta_bootstrap = bootstrap_procedure(y_pred, residuals, beta_estimate, B, steps, time_steps, y, X, bandwidth_2, max_lag)
    q_bootstrap = beta_bootstrap - np.tile(beta_estimate[max_lag], (len(beta_bootstrap), 1))

    theta_estimate = local_linear_estimation([y, X, time_steps, steps, bandwidth_2])
    beta_estimate = theta_estimate[:, 1]


    plot_beta_CI(time_steps[max_lag:], beta_estimate[max_lag:], q_bootstrap, alpha, B, name)


if __name__ == "__main__":
    main()