import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

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

def show_clusters_by_imshow(clusters, verbose=False):
    """
    Show the clusters that are formed during the RG procedure.

    Parameters:
        cluster - ndarray containing the clusters at the different iterations of the RG transformation
        verbose (optional) - if True it prints the cluster list after showing the image 
    """

    original_size = len(clusters[0][0]) * len(clusters[0][:, 0])
    grid = np.zeros((original_size, original_size))
    for c in clusters[:-1]:

        colors = np.arange(1, 1 + len(c[:,0])) / len(c[:,0]) * 10
        for i, color in enumerate(colors):
            for j in c[i]:
                grid[j, c[i]] = color
        
        plt.imshow(grid)
        plt.show()

        if verbose == True:
            print(f"Clusters = {c}")

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
