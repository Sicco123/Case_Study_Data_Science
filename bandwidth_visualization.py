import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np

source = 'cv_bw_selection_ContainmentHealthIndex.pkl' #'cv_bw_selection_GovernmentResponseIndex.pkl' # 'cv_bw_selection_StringencyIndex.pkl'  ##
y_label = 'log-ll' #'AIC'
name = 'log_lik_coantianment_small_steps'

with open(source, 'rb') as f:
    results = pickle.load(f)

for index in indices:

ax.scatter(*zip(*results), s=2)
ax.set_xlabel('Bandwidth')
ax.set_ylabel(y_label)
fig.show()
fig.savefig(f'Figures/{name}_bw.pdf')

    with open(f'bw_selection_{index}.pkl', 'rb') as f:
        results = pickle.load(f)

    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(*zip(*results), linewidth=2)
    ax.set_xlabel('Bandwidth')
    ax.set_ylabel('AIC')
    # ax.set_title(f'{index}')
    fig.tight_layout()
    fig.show()
    fig.savefig(f'Figures/AIC_bw_{index}.pdf')

    bw_values = np.array(list(map(list, zip(*results)))).T
    argmin = np.argmin(bw_values[:,1])
    optimal_bw_row = bw_values[argmin]
    print(f'Optimal bandwidth for {index} is {round(optimal_bw_row[0], 2)} with AIC of {optimal_bw_row[1]}')
