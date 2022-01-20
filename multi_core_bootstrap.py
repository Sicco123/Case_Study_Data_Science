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
import multiprocessing
import time

def simulate_y(y_pred, residuals, B):
    normal_draws = np.random.normal(0,1,size = (B, len(y_pred)))

    repeat_residuals = np.repeat(residuals.reshape(-1,1), B, axis = 1 ).T
    repeat_y_pred = np.repeat(y_pred.reshape(-1,1), B, axis = 1 ).T

    resampled_residuals = normal_draws * repeat_residuals
    #resampled_residuals -= np.mean(resampled_residuals, axis=0)

    simulated_y_pred = repeat_y_pred + resampled_residuals
    return simulated_y_pred

def local_linear_estimation(arguments):
    y, X, time_steps, steps, h = list(list(arguments))
    theta_all = np.empty((len(steps), 2))
    for i, step in enumerate(steps):
        result = ll_estimation.compute_theta(X, y, time_steps, step, h)
        theta_all[i] = result[:2].T

    return theta_all

def plot_beta_CI(x, beta_observed, beta_bootstrap, alpha, B):
    quantile_1 = int(0.5*alpha*(B+1))
    quantile_2 = int(0.5*(1-alpha)*(B+1))

    beta_bootstrap.sort(axis = 0)
    beta_bootstrap_q1 = beta_bootstrap[quantile_1,:]
    beta_bootstrap_q2 = beta_bootstrap[quantile_2,:]

    fig, ax = plt.subplots()
    ax.plot(x, beta_observed, color = 'k')
    ax.fill_between(x, beta_bootstrap_q1, beta_bootstrap_q2, color='r', alpha=.1)
    ax.set_title('Time Varying Estimates', fontweight='bold', fontsize=18)
    fig.savefig('plots/Bootstraped_Confidence_Intervals_Time_Varying_Estimates')

def main():
    bandwith = 0.2
    #steps_size = 1000
    B = 99
    alpha = 0.05

    y, X, time_steps = ll_estimation.get_and_prepare_data()
    steps = time_steps
    steps.sort()

    theta_estimate = local_linear_estimation([y, X, time_steps, steps, bandwith])
    y_pred = np.diagonal(X.T @ theta_estimate.T)
    residuals = y - y_pred
    beta_estimate = theta_estimate[:, 1]


    beta_bootstrap = np.zeros(((B+1),len(beta_estimate)))
    beta_bootstrap[0,:] = beta_estimate
    y_simulated = simulate_y(y_pred, residuals, B)

    #plt.plot(y_simulated[1,:])
    #plt.plot(y)
    #plt.show()

    X_repeat = [X]*B
    time_steps_repeat = [time_steps]*B
    steps_repeat = [steps]*B
    bandwith_repeat = [bandwith]*B
    input = zip(y_simulated, X_repeat, time_steps_repeat, steps_repeat, bandwith_repeat)

    start_time = time.time()
    a_pool = multiprocessing.Pool(processes= 8)
    result = a_pool.map(local_linear_estimation,input)

    for b, res in enumerate(result):
        beta_bootstrap[b+1, :] = res[:,1]
    end_time = time.time()
    print(end_time-start_time)
    #print(beta_bootstrap)
    #for b in range(B):
    #    print(b)
    #    theta_bootstrap_b = ll_estimation.local_linear_estimation(y_simulated[b,:], X, time_steps, steps, bandwith)
    #    beta_bootstrap_b = theta_bootstrap_b[:,1]
    #    beta_bootstrap[(b+1),:] = beta_bootstrap_b


    plot_beta_CI(time_steps, beta_estimate, beta_bootstrap, alpha, B)


if __name__ == "__main__":
    main()