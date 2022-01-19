"""
TESTTTTT
Authors: Jamie Rutgers, Mirte Pruppers, Damiaan Bonnet, Sicco Kooiker
Name: simulate_example1_cai.py
Date: 18-1-2022
Description: In this file we perform a simulation study. We simulate y and X using the function
             "simulate_example_1". Then we choose a bandwidth on the basis of the AIC and for this 
             optimal bandwidth we return the estimate for beta_0 and beta_1 using Local Linear regression. 
"""


from datetime import datetime 
start = datetime.now()
import numpy as np
import math as ms
import matplotlib.pyplot as plt
#from tqdm import tqdm 
from local_linear_estimation_cai import kernel_function, compute_S, compute_T, compute_theta, local_linear_estimation
from bandwidth import compute_W, compute_X_tilde, compute_A, compute_n_h, compute_aic


def beta_0(t):
    """
    function β₀(t)
    """
    return 0.2 * ms.exp(-0.7 + 3.5 * t)

def beta_1(t):
    """
    function β₁(t)
    """
    return 2 * t + ms.exp(-16 * (t - 0.5) ** 2) - 1



time = np.asarray([i for i in range(1,401)])



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
     
    x = np.zeros(n + 1)
    u = np.zeros(n + 1)
    y = np.zeros(n + 1)
    
    for i in range(1, n + 1):
    
        eps_1 = np.random.normal(loc=0,scale =ms.sqrt(2 **(-2)))
        eps_2 = np.random.normal(loc=0,scale =ms.sqrt(4 **(-2)))
        
    
        x[i] = rho_x * x[i-1] + eps_1
        u[i] = rho_u * u[i-1] + eps_2
        
        y[i] = beta_0(i / n) + beta_1(i / n) * x[i] + u[i]
    
    x = np.delete(x,0)
    y = np.delete(y,0)
    
    X = np.vstack((np.ones((1, len(x))), x.reshape(1, -1)))
    time_steps = np.array(time / len(time))
    
    return (y, X, time_steps)



def get_bandwidth_aic(y, X, time_steps, steps):
    """
    get_bandwidth_aic: on the basis of the AIC it returns the optimal bandwidth in "h_opt",
                       but it also returns the theta estimates of LL for this optimal bandwidth in "theta_hopt"
                    y: array of size (n,)   
                    X: array of size (2,n)
           time_steps: array of size (n,)
                steps: array of size (1000,)
    """
    bandwidth_values = [0.1 + i * 0.025 for i in range(25)]
    
    # Lists to store the theta estimates and aics  
    lst_theta_estimate = []
    lst_aic = []
    
    for bw in bandwidth_values:
        
        theta_estimate = local_linear_estimation(y, X, time_steps, steps, bw)
        lst_theta_estimate += [theta_estimate]
        aic = compute_aic(y, X, theta_estimate, time_steps, bw)
        lst_aic = lst_aic + [[bw, aic]]
        
    # h_opt = optimal bandwidth
    # theta_hopt = theta estimates of LL for this optimal bandwidth
    lst_aic  = np.array(lst_aic)
    h_opt = lst_aic[np.where(lst_aic[:, 1] == np.min(lst_aic[:, 1]))[0][0], 0]
    theta_hopt = lst_theta_estimate[np.where(lst_aic[:, 1] == np.min(lst_aic[:, 1]))[0][0]]
    
    return h_opt, theta_hopt



def perform_multiple_LL(number, steps):
    """
    perform_multiple_LL: simulates "number"-times data; for each simulation dataset we 
                         get the optimal bandwidth and its corresponding estimates of beta_0 and beta_1
                         using "get_bandwidth_aic" function. The results are stored and returned in a list "store"
                number:  integer 
                steps :  array of size (1000,)
                         
    """
    store = []
    h_store = []
    for i in range(number):
        
        y, X, time_steps = simulate_example_1()
        h_opt, theta_hopt = get_bandwidth_aic(y, X, time_steps, steps)
        store += [theta_hopt]
        h_store += [h_opt]
        
    return store, h_store   




def main():
    # steps = np.random.uniform(low = 0, high = 1, size=(1000, ))
    # steps.sort()
    
    # store_theta_estimate = perform_multiple_LL(2, steps)
    
    # p_i = lambda t: beta_1(t) 
    # vfun = np.vectorize(p_i, otypes=[float])
      
    # for ar in store_theta_estimate:
    #      plt.figure()
    #      plt.plot(steps,ar[:,1])
    #      plt.plot(steps,vfun(steps))  
    print("niks")

if __name__ == '__main__':
    steps = np.random.uniform(low = 0, high = 1, size=(1000, ))
    steps.sort()
    
    store_theta_estimate, h_store = perform_multiple_LL(2, steps)
    
    p_i = lambda t: beta_1(t) 
    vfun = np.vectorize(p_i, otypes=[float])
      
    for ar in store_theta_estimate:
          plt.figure()
          plt.plot(steps,ar[:,1])
          plt.plot(steps,vfun(steps))  

