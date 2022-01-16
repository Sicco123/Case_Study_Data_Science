"""
Authors: Jamie Rutgers, Mirte Pruppers, Damiaan Bonnet, Sicco Kooiker
Name: static_coefficient_estimation.py
Date: 12-1-2022
Desription: In this file the COVID-19 measures data are regressed on the COVID-19
reproduction number data. We use a time-varying coefficient model,
in which we estimate the coefficient functions using local linear regression.
Besides we return the inference results. For the Confidence Intervals we use the methods
programmed in "inference_bootstrap.py". As bandwith we use methods from "bandwith_selection.py".
"""
import statsmodels.api as sm
import numpy as np
import pickle
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time


def ols_regression(y, X):
    """
    y   series/array : Reproduction number data
    X   DataFrame : Regressors
    -----------
    results resultobject
    ___________
    This methods performs OLS regression and returns the results
    """
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results

def kernel_function(u):
    """
    kernel_function: returns the function value of the Epanechnikov kernel with u as the argument
                  u: float
    """
    res = np.zeros(len(u))

    kernel_array = (3 / 4) *(1 - u ** 2)
    indices_1 = np.where(np.abs(u) < 1)
    res[indices_1] = kernel_array[indices_1]

    return res


def tv_ll_regression(y, X, steps, h):
    """
    y   series/array : Reproduction number data
    X   DataFrame : Regressors
    -----------
    tv_ll: returns all the local linear estimators which is an nx2x1 array
    -----------
    This methods performs time varying local linear regression and returns the results
    """

    t_i = steps

    # create (n x n)-matrix K
    k_i = lambda t: kernel_function((t_i - t) / h)
    X_reshaped = X.reshape((len(X),1))

    K = np.apply_along_axis(k_i, 1, X_reshaped).T
    K = np.apply_along_axis(np.diag, 1, K) # every [i,0:N,0:N] is the appropriate K matrix for beta(i)

    # create Z
    h_func = lambda m: X - m
    z_func = lambda m: np.hstack((np.ones((len(X), 1)), h_func(m).reshape((len(X), 1))))

    t_i_reshaped = t_i.reshape((len(t_i), 1))
    Z = np.apply_along_axis(z_func, 1, t_i_reshaped)

    # Do matrix multiplications to get our estimator = (Z'KZ)⁻¹Z'Ky
    M1 = np.linalg.inv(np.matmul(Z.transpose((0,2,1)), np.matmul(K, Z)))  # M1  = (Z'KZ)⁻¹
    M2 = np.matmul(Z.transpose((0,2,1)), K)  # M2  =  Z'K

    res = np.matmul(np.matmul(M1, M2), y)  # res =  M1 * M2 * y

    return res

def transform_data(y, X):
    """
    y   series/array : Reproduction number data
    X   DataFrame : Regressors
    -----------
    y_log   series/array : log of y
    X_log   DataFrame : log of X
    ___________
    This method returns the log of the given input.
    """
    y_log = np.log(y)
    X_log = np.log(X)
    return y_log, X_log

def fit_sin_simulation():
    # initialisation
    # simulate n = 200 datapoints. x = (x_1, ..., x_n)' and y = (y_1, ..., y_n)'
    X = np.random.uniform(low=0, high=6, size=(200,))  # x_i ~ Uniform(0,6)
    y = np.sin(X) + np.random.normal(scale=0.5, size=(200,))  # y_i ~ N(sin(x_i), 0.5)

    steps = np.random.uniform(low=0, high=6, size=(600,))  # x_i ~ Uniform(0,6)
    start_time = time.process_time()
    steps.sort()
    results = tv_ll_regression(y, X, steps, 1)
    end_time = time.process_time()

    print(f'The local linear regression takes {end_time-start_time}')
    # results
    plot_results(np.sin(steps), results[:,0], steps, X, y, 'Sin simulation')


def plot_results(y, y_pred, X, X_sim, y_sim, index_name):
    """
    y   series/array : Reproduction number data
    X   DataFrame : Regressors
    -----------
    results resultobject
    ___________
    This methods performs OLS regression and returns the results
    """

    plt.figure()
    plt.title(f'{index_name} time varying', fontweight='bold', fontsize=18)
    plt.plot(X, y)
    plt.plot(X, y_pred)
    plt.scatter(X_sim, y_sim, [3] * len(y_sim))  # plot simulated data points

    #plt.savefig(f'plots/{index_name}_time_varying_regression')
    plt.show()


def main(simulation=True):
    # read data
    source = 'reproduction_vs_index_Japan.pkl'
    with open(source, 'rb') as f:
        raw = pickle.load(f)

    data = raw.dropna()  # removes nans
    regressor_names = ['StringencyIndex']#, 'GovernmentResponseIndex', 'ContainmentHealthIndex']

    if simulation:
        fit_sin_simulation()
    else:
        y_log, X_log = transform_data(data['Rt'], data['StringencyIndex'])
        y_log, X_log = np.array(y_log), np.array(X_log)
        steps = np.array(data['time_index']/len(data))
        results = tv_ll_regression(y_log, X_log, steps, 1)
        plot_results(X=data['Date'],
                     y_pred=results[:,1],
                     X_sim=None,
                     y_sim = None,
                     index='StringencyIndex')


    # for index in regressor_names:



if __name__ == "__main__":
    main(simulation=False)