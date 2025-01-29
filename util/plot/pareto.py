import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import glob, os, gc
from tqdm import tqdm

def pareto_front(complexity, error):
    """
    Find the Pareto front for minimizing both complexity and error.

    Parameters:
        complexity (list or array): Number of unique variables (complexity) for each model.
        error (list or array): Validation RMSE of each model.

    Returns:
        pareto_complexity, pareto_error: Arrays containing points on the Pareto front.
    """
    # Sort by complexity, then by error (both ascending)
    sorted_indices = np.lexsort((error, complexity))
    complexity = np.array(complexity)[sorted_indices]
    error = np.array(error)[sorted_indices]
    
    pareto_complexity = []
    pareto_error = []
    
    # Track the lowest RMSE seen so far
    min_error = float("inf")
    
    for c, e in zip(complexity, error):
        if e < min_error:  # Only add if it improves RMSE
            pareto_complexity.append(c)
            pareto_error.append(e)
            min_error = e
    
    return np.array(pareto_complexity), np.array(pareto_error)

def _get_mean_RMSE(storeper,valretain,exptype):
    fourierlv = [3,5,7,9,11,13]
    store = []
    for i in range(6):
        TESTlist = []
        for j in range(len(storeper[valretain][f'F_{fourierlv[i]}'])):
            TESTlist.append(np.asarray(storeper[valretain][f'F_{fourierlv[i]}'][j][f'{exptype}_rmses']).mean())
        store.append(TESTlist)
    return store

def plot_pareto(store_vars, store_vars_f, exptype, varexps, climatology, figsize, ylabel, title, savepath, legend, xlim):
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)  # Slightly larger figure for clarity
    alphas = [0.3, 0.5, 0.7, 0.9, 1.0]  # Example alpha values for illustration

    # Loop through variance explained and scatter plot
    colors = {
        '95': 'blue', '85': 'red', '90': 'black',
        '80': 'purple', '75': 'orange', '99': 'green'
    }
    markers = {
        '95': 'o', '85': 's', '90': '^',
        '80': 'D', '75': 'v', '99': 'P'
    }

    for varexp in varexps: #colors.keys():
        for i in range(5):
            ax.scatter(
                store_vars[varexp]['numvars'][i],
                store_vars_f[varexp][f'{exptype}_rmses'][i],
                color=colors[varexp],
                marker=markers[varexp],
                s=80,
                alpha=alphas[i]
            )
        ax.scatter(
            store_vars[varexp]['numvars'][-1],
            store_vars_f[varexp][f'{exptype}_rmses'][-1],
            color=colors[varexp],
            marker=markers[varexp],
            s=120,
            label=f'{varexp}% variance explained',
            edgecolor='k'
        )

    # Add horizontal line for climatology
    ax.axhline(climatology, ls='--', lw=2, color='tab:red', label='Climatology')

    # Customize labels and title
    ax.set_xlabel('Number of Unique Features', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(xlim[0],xlim[1])

    # Add grid for better readability
    ax.grid(lw=1.2, ls='--', alpha=0.3)

    # Add legend
    if legend == 'Yes':
        ax.legend(loc='upper right', fontsize=10, frameon=True)
    elif legend == 'No':
        pass

    # Adjust ticks
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Set the spine thickness
        
    # Show plot
    plt.tight_layout()
    plt.savefig(savepath,dpi=600)
    plt.show()
    
    return fig, ax

def plot_pareto_convertedCDF(store_vars, store_varsexpdict_cdf2max, exptype, varexps, climatology, figsize, ylabel, title, savepath, legend, ylim):
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)  # Slightly larger figure for clarity
    alphas = [0.3, 0.5, 0.7, 0.9, 1.0]  # Example alpha values for illustration

    # Loop through variance explained and scatter plot
    colors = {
        '95': 'blue', '85': 'red', '90': 'black',
        '80': 'purple', '75': 'orange', '99': 'green'
    }
    markers = {
        '95': 'o', '85': 's', '90': '^',
        '80': 'D', '75': 'v', '99': 'P'
    }

    for varexp in varexps: #colors.keys():
        for i in range(5):
            ax.scatter(
                store_vars[varexp]['numvars'][i],
                _get_mean_RMSE(store_varsexpdict_cdf2max,varexp,exptype)[i],
                color=colors[varexp],
                marker=markers[varexp],
                s=80,
                alpha=alphas[i]
            )
        ax.scatter(
            store_vars[varexp]['numvars'][-1],
            _get_mean_RMSE(store_varsexpdict_cdf2max,varexp,exptype)[-1],
            color=colors[varexp],
            marker=markers[varexp],
            s=120,
            label=f'{varexp}% variance explained',
            edgecolor='k'
        )

    # Add horizontal line for climatology
    ax.axhline(climatology, ls='--', lw=2, color='tab:red', label='Climatology')

    ax.set_ylim(ylim[0],ylim[1])
    # Customize labels and title
    ax.set_xlabel('Number of Unique Features', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)

    # Add grid for better readability
    ax.grid(lw=1.2, ls='--', alpha=0.3)

    # Add legend
    if legend == 'Yes':
        ax.legend(loc='upper right', fontsize=10, frameon=True)
    elif legend == 'No':
        pass

    # Adjust ticks
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Thicken spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Set the spine thickness
        
    # Show plot
    plt.tight_layout()
    plt.savefig(savepath,dpi=600)
    plt.show()
    
    return fig, ax