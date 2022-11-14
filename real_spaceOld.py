import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from data_cleaning import *
from scipy.optimize import minimize
from scipy.stats import rv_histogram
from matplotlib.lines import Line2D
import math


def real_space_rg_iteration(X, correlation, test=False):
    """
    Perform a single RG iteration. Given the dataset X and its correlation matrix we greedily pair the 
    highest correlated pairs with each other. 

    Parameters:
        X - 2d np array containing the data
        correlation - 2d np.array containing the correlation matrix of the data
        test (optional) - if set to True it runs a simple test on a 2 by 2 lattice system. See output for more information.

    Return:
        X_coarse - the coarse grained variables. Size = [len(X)/2, len(X)/2]
        pairings - list of indices that were paired. 
    """
    # Initialize
    X_coarse = np.zeros((X.shape[0]//2, X.shape[1]))
    pairings = []
    list_of_original_indices = np.array([np.arange(len(X)), np.arange(len(X))])

    # Interested in absolute correlation
    correlation = np.abs(correlation)
    np.fill_diagonal(correlation, 0)

    if test == True:
        correlation = np.array([[0, 0, 7, 0], [0, 0, 0, 3], [7, 0, 0, 0], [0, 3, 0, 0]])
        X = np.random.randint(2, size=(4, 3))
        X_coarse = np.zeros((X.shape[0]//2, X.shape[1]))
        print("Testing the algorithm for a simple case.\n#######################################")
        print(f"The correlation matrix is given by:\n{correlation}")
        print(f"Original dataset X:\n{X}")


    for i in range(len(correlation) // 2):
        # Find highest correlated pair from correlation matrix
        max_idx = correlation.argmax()
        max_i, max_j = max_idx // len(correlation), max_idx % len(correlation)

        if max_i == max_j:
            print('found diagonal element')
            print(correlation)

        # Remove the corresponding row and column from correlation matrix
        correlation = np.delete(correlation, [max_i, max_j], axis=0)
        correlation = np.delete(correlation, [max_i, max_j], axis=1)

        # Save found pairing
        max_i_original = list_of_original_indices[0][max_i]
        max_j_original = list_of_original_indices[1][max_j]

        pairings.append([max_i_original, max_j_original])
        list_of_original_indices = np.delete(list_of_original_indices, [max_i, max_j, max_i + len(list_of_original_indices[0]), max_j + len(list_of_original_indices[0])])

        # np.delete reshapes the array, undo this
        if len(list_of_original_indices) != 0:
            list_of_original_indices = list_of_original_indices.reshape(-1, len(correlation))
        elif len(list_of_original_indices) == 1:
            pairings.append(list_of_original_indices[0])

        # Merge pair in dataset also
        X_coarse[i] = (X[max_i_original] + X[max_j_original]) / 2

    if test == True:
        print("\nResults\n#######################################")
        print(f"Pairings found = {pairings}")
        print(f"Coarse grained dataset:\n{X_coarse}\n")

    return X_coarse, pairings

def compute_couplings(X):
    """
    Finds the best fit for the coupling of the full model with X as dataset.

    Parameters:
        X - dataset to fit with

    Return:
        couplings - list of couplings
    """
    # Create pd for complete model
    f1 = np.array([[1,1],[1,-1]])
    f2 = np.kron(f1,f1)
    f4 = np.kron(f2,f2)
    f8 = np.kron(f4,f4)
    n = 8

    # Data average 
    X_df = pd.DataFrame(X.T)
    print(X_df.shape)
    n_rows, n_cols = X_df.shape[0], X_df.shape[1]
    cols = list(range(n_cols))
    X_df = X_df.groupby(cols)[0].count()

    X_unique = X_df.index.tolist()
    X_counts = X_df.values.tolist()
    states_strings = [''.join([str(s) for s in state]) for state in X_unique]
    data_int = [int(s,2) for s in states_strings]
    g_data = np.zeros(2**n)
    g_data[data_int] = np.array(X_counts) / sum(X_counts)
    print(data_int, X_counts)

    # Initial parameters
    g = np.random.rand(2**n)
    
    epsilon = 0.1

    N = 1000
    for i in range(1):
        _, dkl_grad = dkl(g, f8, g_data)
        g += dkl_grad
    return couplings

def dkl(g, f, g_data):
    # Complete model probability distribution
    z = np.sum(np.exp(np.dot(f, g)))
    pd_complete_model = np.exp(np.dot(f, g)) / z

    # Data probability distribution
    z_data = np.sum(np.exp(np.dot(f, g_data)))
    pd_data = np.exp(np.dot(f, g)) / z_data

    l = np.log(pd_complete_model / pd_data)
    l[np.isinf(l)] = 0 
    l[np.isnan(l)] = 0
    dkl = np.dot(pd_complete_model, l)
    dkl_grad = np.dot(f.T, pd_data-pd_complete_model)
    #print(dkl_grad)
    return dkl, dkl_grad

def add_to_clusters(clusters, pairings):
    """
    Add pairings found at a RG iteration to clusters that have already been formed by
    the previous iterations.

    Parameters:
        clusters - 2darray of non-coarse grained variables per cluster
        pairing - pairings of variables found at a RG iteration

    Return:
        clusters - 2darray containing new clusters. 
    """
    # First RG iteration
    if len(clusters) == 0:
        return pairings

    # Loop over pairings found and create new clusters
    new_clusters = []
    for _, pair in enumerate(pairings):
        if len(pair) == 1: # This variable was not paired
            new_cluster = pair
        elif len(pair) == 2:
            new_cluster = np.array([clusters[pair[0]], clusters[pair[1]]])
        else:
            print("Found a pair with length > 2. Something went wrong.")
       
        # Reshape clusters so it stays a 2d array
        new_clusters.append(new_cluster.reshape(-1))
    return new_clusters

def real_space_rg(X, steps, test=False):
    """
    """
    X_list = []
    clusters_list = []
    clusters = []
    X_coarse = X
    coupling_parameters = []

    # Perform RG iterations
    for i in range(steps):
        # Cannot coarse any further
        if len(X_coarse) == 1:
            print("Finished coarse-graining!")
            return np.array(X_list), np.array(clusters_list), np.array(coupling_parameters)

        # Compute couplings
        # coupling_parameters.append(compute_couplings(X_coarse))

        # Compute correlation
        correlation = np.corrcoef(X_coarse)

        # Save data
        X_list.append(X_coarse)

        # RG iteration
        X_coarse, pairings = real_space_rg_iteration(X_coarse, correlation, test=test)

        # Add pairing to clusters
        clusters = add_to_clusters(clusters, pairings)
        clusters_list.append(np.array(clusters))

    print("Finished coarse-graining!")
    return np.array(X_list), np.array(clusters_list), np.array(coupling_parameters)

def show_clusters_by_imshow(clusters):
    """
    """

    original_size = len(clusters[0][0]) * len(clusters[0][:, 0])
    grid = np.zeros((original_size, original_size))
    for c in clusters[:-1]:

        colors = np.arange(1, 1 + len(c[:,0])) / len(c[:,0]) * 10
        for i, color in enumerate(colors):
            for j in c[i]:
                grid[j, c[i]] = color

        print(f"Clusters = {c}")
        plt.imshow(grid)
        plt.show()

def plot_normalized_activity(X_list):
    """
    """
    n_vars = X_list[0].shape[0] # Variables in dataset before coarse-graining
    bins = 50

    figs, axs = plt.subplots(1)
    markers = list(Line2D.markers.keys())[3:len(X_list) + 3]

    for i, X in enumerate(X_list):
        # Create histogram
        hist_values, hist_edges = np.histogram(X, bins=bins, density=True)
        hist_edges = [(hist_edges[i] + hist_edges[i+1]) / 2 for i in range(len(hist_edges) - 1)]

        # Plot histogram
        axs.plot(hist_edges, hist_values, marker=markers[i], markersize=5, label=f"K = {n_vars // X.shape[0]}")

    plt.ylabel("probability")
    plt.xlabel("normalized activity")
    plt.title("Probability distribution of the normalized activity")
    plt.legend()
    plt.show()

def plot_eigenvalue_scaling(X_coarse):
    """
    Plot the eigenvalues spectrum of the correlation function at different steps of the 
    coarse graining.

    Parameters:
        X_coarse - a list of arrays containing the activity of the orignal and coarse-grained variables. 
    """
    n_vars = X_coarse[0].shape[0]
    for X in X_coarse:
        # Compute correlation matrix
        corr = np.corrcoef(X)

        # Compute its eigenvalues
        eigvalues, eigvectors = np.linalg.eig(corr)

        # Plot spectrum
        sort_idx = np.argsort(eigvalues)
        eigvalues = eigvalues[sort_idx][::-1]
        rank = np.arange(0, len(eigvalues)) / len(eigvalues)
        plt.plot(rank, eigvalues, "o--", label=f"K = {n_vars // X.shape[0]}")
    
    plt.ylabel("eigenvalues")
    plt.xlabel("rank/K")
    plt.legend()
    plt.show()

def plot_n_largest_eigenvectors(X_coarse, n):
    """
    Plot the n largest eigenvectors in an imshow figure.

    Parameters:
        X_coarse - a list of arrays containing the activity of the orignal and coarse-grained variables. 
        n - number of eigenvectors to plot
    """

    corr = np.corrcoef(X_coarse[0])
    eigvalues, eigvectors = np.linalg.eig(corr)

    plot_size = math.ceil(np.sqrt(n))
    fig, axs = plt.subplots(plot_size, plot_size)
    for i in range(n):
        row, col = i // plot_size, i % plot_size
        eigvector = eigvectors[i].reshape(-1, 1)
        im = axs[row, col].imshow(eigvector @ eigvector.T)
        axs[row, col].set_ylabel("eigenvalues")
        axs[row, col].set_xlabel("rank/K")
        axs[row, col].set_title(f"Eigenvector {i+1}")
        fig.subplots_adjust(hspace=.8)
        fig.colorbar(im)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Read input 
    X = read_input()
    X = check_dataset(X.T)
    print(f"Shape of input data = {X.shape}")

    steps = 5
    test = False    
    X_coarse, clusters, coupling_parameters = real_space_rg(X, steps, test=test)

    # show_clusters_by_imshow(clusters)

    plot_normalized_activity(X_coarse)

    plot_eigenvalue_scaling(X_coarse)

    n = 4
    # plot_n_largest_eigenvectors(X_coarse, n)