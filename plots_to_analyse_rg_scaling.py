import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

def plot_normalized_activity(p_averages, p_stds, unique_activity_values):
    """
    Plots the distribution of the normalized activity. Given the average probabilities, standard deviations and the unique values at each
    step of the coarse-graining. This data can be obtained from the RG_class.

    Parameters:
        p_averages - a list of size = n_rg_iterations with each list containing an numpy array of the average activity across all clusters
        p_stds - a list of size = n_rg_iterations with each list containing an numpy array of the standard deviation of the activity across all clusters
        unique_activity_values - a list of size = n_rg_iterations with each list containing an numpy array of the unique activity values in the clusters
    """
    for i, _ in enumerate(p_averages):
        cluster_size = np.log2(len(p_averages[i]) - 1) + 1
        plt.errorbar(unique_activity_values[i], p_averages[i], 2*p_stds[i], fmt="o--", markersize=3, label=f"K = {cluster_size}")

    plt.ylabel("probability")
    plt.xlabel("normalized activity")
    plt.title("Probability distribution of the normalized activity")
    plt.grid(True)
    plt.legend(loc="upper right")
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
        rank = np.arange(1, len(eigvalues)+1) / len(eigvalues)
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

def plot_eigenvalue_spectra_within_clusters(Xs, clusters):
    """
    This function plots the eigenvalue spectra within the clusters. At each coarse-grained level the mean and variance of the spectra
    across the different clusters are computed and plotted.

    Parameters:
        Xs - list contianing the dataset at each coarse-grained level
        clusters - list containing the clusters that where formed at the different coarse-grianing iterations
    """
    original_dataset = Xs[0]

    # Loop over coarse-graining iterations
    for i, cluster in enumerate(clusters):
        
        # Compute cluster size
        cluster_size = len(clusters[0]) // len(cluster) * 2
        
        # Not interested in the spectra of these small clusters
        if cluster_size <= 2:
            continue
            
        # Compute the spectrum for each cluster, average and plot with std
        eigvalues_l = []
        for c in cluster:
            corr = np.corrcoef(original_dataset[c])
            eigvalues, _ = np.linalg.eig(corr)
            eigvalues_l.append(np.sort(eigvalues)[::-1])
            
        # Compute statistics
        rank = np.arange(1, len(eigvalues) + 1) / len(eigvalues)
        mean = np.mean(eigvalues_l, axis=0)
        std = np.std(eigvalues_l, axis=0)
        
        # Plot
        plt.errorbar(rank, mean, 3*std, fmt="o--", label=f"K = {cluster_size}")
        
    plt.xlabel("rank/K")
    plt.ylabel("eigenvalues")
    plt.legend()
    plt.show()

def plot_free_energy_scaling(p_averages, p_stds, unique_activity_values):
    """
    When a RG transformation is exact the free energy does not change. This function compute the free energy at each
    coarse-grained step and log plots the values. We hope to see some scaling with a power law close to 1.

    We can compute the free energy by F = -np.log(p0) with p0 the probability that a cluster is silent.

    Parameters:
        X_list - nd numpy array containing the variables at different steps of the coarse-graining
    """
    p0_avg = []
    p0_std = []
    cluster_sizes = []
    for i, unique_vals in enumerate(unique_activity_values):
        # Find idx at which cluster is silent
        idx = np.argwhere(unique_vals == 0.0)
        
        # Check it exists
        if len(idx) != 0:
            idx = idx[0]
            
            # Add to list
            p0_avg.append(list(p_averages[i][idx])[0])
            p0_std.append(list(p_stds[i][idx])[0])

            # Compute cluster size
            cluster_size = np.log2(len(p_averages[i]) - 1) + 1
            cluster_sizes.append(cluster_size)

    # Plot the probability of the cluster being silent
    plt.errorbar(cluster_sizes, p0_avg, 3*np.array(p0_std), fmt="g^--")
    plt.xlabel("cluster size")
    plt.ylabel(r"ln P$_{Silence}$")
    plt.yscale("log")
    plt.grid(True)
    plt.show()
