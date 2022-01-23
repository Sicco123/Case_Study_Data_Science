"""
Authors: Jamie Rutgers, Mirte Pruppers, Damiaan Bonnet, Sicco Kooiker
Name: simulate_example1_cai.py
Date: 18-1-2022
Description: In this file we perform a simulation study. We simulate y and X using the function
             "simulate_example_1". Then we choose a bandwidth on the basis of the AIC and for this 
             optimal bandwidth we return the estimate for beta_0 and beta_1 using Local Linear regression. 
"""


from datetime import datetime 
import pickle
import numpy as np
import math as ms
import matplotlib.pyplot as plt
from local_linear_estimation_cai import kernel_function, compute_S_k, compute_T, compute_theta, local_linear_estimation
from bandwith_selection import compute_aic
from tqdm import tqdm

import pandas as pd

def beta_0(t):
    """ function β₀(t) """
    return 0.2 * ms.exp(-0.7 + 3.5 * t)

def beta_1(t):
    """function β₁(t)"""
    return 2 * t + ms.exp(-16 * (t - 0.5) ** 2) - 1

vbeta_1 = np.vectorize(beta_1, otypes=[float])


def simulate_example_1():
    """
    see example 1 Cai 2007
    DGP: Yᵢ = β₀(tᵢ) + β₁(tᵢ)Xᵢ + uᵢ
    
    β₀(t) = 0.2 exp(-0.7 + 3.5t)
    β₁(t) = 2t + exp(-16(t - 0.5)²) - 1
    """
    n = 400
    rho_x = 0.9
    rho_u = 0.8
     
    x, u, y  = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    
    for i in range(1, n + 1):
    
        eps_1 = np.random.normal(loc=0,scale =ms.sqrt(2 **(-2)))
        eps_2 = np.random.normal(loc=0,scale =ms.sqrt(4 **(-2)))
        
    
        x[i] = rho_x * x[i-1] + eps_1
        u[i] = rho_u * u[i-1] + eps_2
        
        y[i] = beta_0(i / n) + beta_1(i / n) * x[i] + u[i]
    
    x = np.delete(x,0)
    y = np.delete(y,0)
    
    X = np.vstack((np.ones((1, len(x))), x.reshape(1, -1)))
    time = np.asarray([i for i in range(1, n + 1)])
    time_steps = np.array(time / len(time))
    
    return (y, X, time_steps)



def get_bandwidth_aic(y, X, time_steps, steps):
    """
    get_bandwidth_aic: on the basis of the AIC it returns the optimal bandwidth in "h_opt",
                        but it also returns the theta estimates of LL for this optimal bandwidth
                        in "theta_hopt".
                    y: array of size (n,)   
                    X: array of size (2,n)
            time_steps: array of size (n,)
                steps: array of size ?
    """
    # bandwidth_values = [0.01 + i * 0.0025 for i in range(4)]
    bandwidth_values = np.linspace(0.01, 0.1, 10)

    # Lists to store the theta estimates and aics  
    lst_theta_estimate = []
    lst_aic = []
    
    for bw in tqdm(bandwidth_values):
        theta_estimate = local_linear_estimation(y, X, time_steps, steps, bw)
        y_pred = np.diagonal(np.matmul(X.T, theta_estimate.T))
        lst_theta_estimate += [theta_estimate]
        aic = compute_aic(y, X, y_pred, time_steps, bw)
        print(bw, aic)
        lst_aic += [[bw, aic]]
        
    # h_opt = optimal bandwidth
    # theta_hopt = theta estimates of LL for this optimal bandwidth
    lst_aic  = np.array(lst_aic)
    h_opt = lst_aic[np.where(lst_aic[:, 1] == np.min(lst_aic[:, 1]))[0][0], 0]
    theta_hopt = lst_theta_estimate[np.where(lst_aic[:, 1] == np.min(lst_aic[:, 1]))[0][0]]
    
    return h_opt, theta_hopt


def perform_multiple_LL(number):
    """
    perform_multiple_LL: simulates "number"-times data; for each simulation dataset we 
                         get the optimal bandwidth and its corresponding estimates of 
                         beta_0 and beta_1  using "get_bandwidth_aic" function. The 
                         results are stored and returned in a list "store".
                number:  integer 
                         
    """
    store = []
    h_store = []
    for i in range(number):
        y, X, time_steps = simulate_example_1()
        h_opt, theta_hopt = get_bandwidth_aic(y, X, time_steps, time_steps)
        store += [theta_hopt]
        h_store += [h_opt]
        
    return store, h_store   



def do_LL_for_specific_bw(number, bw):
    """
    do_LL_for_specific_bw: Returns a specific
                
    """
    store_theta = []
    store_timesteps = []
    store_made = []
    
    for i in range(number):
        y, X, time_steps = simulate_example_1()
        theta_estimate = local_linear_estimation(y, X, time_steps, time_steps, bw)
        store_theta += [theta_estimate]
        store_timesteps += [time_steps]
        store_made += [MADE(theta_estimate[:,1], vbeta_1(time_steps))]
        
    return store_theta, store_timesteps, store_made


def MADE(beta_hat, beta):
    """MADE: calculates the MADE which is defined as MADE = ..."""
    return np.sum(abs(beta_hat - beta)) / len(beta_hat)
    

def main():
    
    # curve_estimates_1, lst_timesteps, lst_made = do_LL_for_specific_bw(50, 0.275)
    curve_estimates_2, h_store = perform_multiple_LL(1)

    print(h_store)
    
    # plt.figure()
    # for curve, time_steps in zip(curve_estimates_1, lst_timesteps):
    #     plt.plot(time_steps, curve[:,1], color = "green")
    #     plt.plot(time_steps, vbeta_1(time_steps), color = "red")
    #
    # return lst_made

if __name__ == '__main__':
    start = datetime.now()
    result = main()
    print(datetime.now() - start)