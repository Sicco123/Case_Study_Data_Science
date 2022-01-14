import pickle
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
import pandas as pd
#%% Import data
with open('reproduction_vs_index_Japan.pkl', 'rb') as f:
    raw = pickle.load(f)


#%% remove rows with NA 
data = raw.dropna()

# Regress Rt for three indices 

for index in ["StringencyIndex",'GovernmentResponseIndex', "ContainmentHealthIndex"]:
    
    # OLS regression
    X = np.log(data[index])
    X = sm.add_constant(X)
    Y = np.log(data['Rt'])
    print(len(Y))
    print(len(X))
    model = sm.OLS(Y,X)
    results = model.fit()
    print(results.params)

    # Plot 
    plt.figure()
    plt.scatter(X[index], Y, [2] * len(Y) )
    plt.plot(X[index], results.params[1] * X[index] + results.params[0])

    plt.show()