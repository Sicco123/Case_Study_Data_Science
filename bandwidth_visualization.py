import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np

indices = ['StringencyIndex', 'GovernmentResponseIndex', 'ContainmentHealthIndex']
matplotlib.rcParams.update({'font.size': 12})


for index in indices:


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

