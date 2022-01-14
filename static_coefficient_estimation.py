"""
Authors: Jamie Rutgers, Mirte Pruppers, Damiaan Bonnet, Sicco Kooiker
Name: static_coefficient_estimation.py
Date: 12-1-2022
Desription: In this file the COVID-19 measures data are regressed on the COVID-19
reproduction number data. We use a simple OLS method. Besides we return the inference results.
"""
import statsmodels.api as sm
import numpy as np
import pickle
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


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

def plot_results(y, X, params, index_name):
    """
    y   series/array : Reproduction number data
    X   DataFrame : Regressors
    -----------
    results resultobject
    ___________
    This methods performs OLS regression and returns the results
    """
    pred = params[1] * X.values + params[0]
    plt.figure()
    plt.title(f'{index_name} OLS regression', fontweight='bold', fontsize=18)
    plt.scatter(X, y, [2] * len(y))
    plt.plot(X.values, pred)
    plt.savefig(f'plots/{index_name}_OLS_regression')
    plt.show()

def main():
    # read data
    source = 'reproduction_vs_index_Japan.pkl'
    with open(source, 'rb') as f:
        raw = pickle.load(f)

    data = raw.dropna()    #removes nans
    regressor_names = ['StringencyIndex', 'GovernmentResponseIndex', 'ContainmentHealthIndex']

    for index in regressor_names:
        # estimation
        y, X = transform_data(data['Rt'], data[index])

        results = ols_regression(y, X)

        # results
        plot_results(y, X, results.params, index)

if __name__ == "__main__":
    main()