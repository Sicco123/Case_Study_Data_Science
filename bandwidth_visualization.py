import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('bw_selection_2002.pkl', 'rb') as f:
    results = pickle.load(f)

plt.scatter(*zip(*results), s=0.5)
plt.show()

bw_values = np.array(list(map(list, zip(*results)))).T
argmin = np.argmin(bw_values[:,1])
optimal_bw_row = bw_values[argmin]
print(f'Optimal bandwidth for 2007 based selection is {round(optimal_bw_row[0], 2)} with AIC of {optimal_bw_row[1]}')

with open('bw_selection_2007.pkl', 'rb') as f:
    results = pickle.load(f)

plt.scatter(*zip(*results), s=0.5)
plt.show()

bw_values = np.array(list(map(list, zip(*results)))).T
argmin = np.argmin(bw_values[:,1])
optimal_bw_row = bw_values[argmin]
print(f'Optimal bandwidth for 2002 based selection is {round(optimal_bw_row[0], 2)} with AIC of {optimal_bw_row[1]}')

