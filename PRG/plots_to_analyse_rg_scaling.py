import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
from scipy.stats import binom
import matplotlib.colors as mcolors

def plot_normalized_activity(p_averages, p_stds, unique_activity_values, clusters, rg_range=(0,0), title=""):
    """
    Plots the distribution of the normalized activity. Given the average probabilities, standard deviations and the unique values at each
    step of the coarse-graining. This data can be obtained from the RG_class.

    Parameters:
        p_averages - a list of size = n_rg_iterations with each list containing an numpy array of the average activity across all clusters
        p_stds - a list of size = n_rg_iterations with each list containing an numpy array of the standard deviation of the activity across all clusters
        unique_activity_values - a list of size = n_rg_iterations with each list containing an numpy array of the unique activity values in the clusters
    """
    cluster_sizes = [len(clusters[0]) / len(c) for c in clusters]
    
    if rg_range != (0,0):
        cluster_sizes = cluster_sizes[rg_range[0]:rg_range[1]]
        p_averages = p_averages[rg_range[0]:rg_range[1]]
        p_stds = p_stds[rg_range[0]:rg_range[1]]
        unique_activity_values = unique_activity_values[rg_range[0]:rg_range[1]]
        clusters = clusters[rg_range[0]:rg_range[1]]

    for i, _ in enumerate(p_averages):
        plt.errorbar(unique_activity_values[i], p_averages[i], 2*p_stds[i], fmt="o--", markersize=3, label=f"K = {cluster_sizes[i]}")

        # Plot probability distribution of binomial p=0.5 with n=cluster_size
        n = cluster_sizes[i]
        p = 0.5
        x = np.arange(0, n+1)
        plt.plot(x / cluster_sizes[i], binom.pmf(x, n, p), '--', color="grey", alpha=0.3)
        

    plt.ylabel("probability")
    plt.xlabel("normalized activity")
    plt.title("Probability distribution of the normalized activity")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.title(title)
    plt.show()

def show_clusters_by_imshow(clusters, verbose=False, title=""):
    """
    Show the clusters that are formed during the RG procedure.

    Parameters:
        cluster - ndarray containing the clusters at the different iterations of the RG transformation
        verbose (optional) - if True it prints the cluster list after showing the image 
    """

    original_size = len(clusters[0][0]) * len(clusters[0][:, 0])
    grid = np.zeros((original_size, original_size))
    for c in clusters:

        colors = np.arange(1, 1 + len(c[:,0])) / len(c[:,0]) * 10
        for i, color in enumerate(colors):
            for j in c[i]:
                grid[j, c[i]] = color
        
        plt.imshow(grid)
        plt.title(title)
        plt.show()

        if verbose == True:
            print(f"Clusters = {c}")

def plot_eigenvalue_scaling(X_coarse, clusters, title=""):
    """
    Plot the eigenvalues spectrum of the correlation function at different steps of the 
    coarse graining.

    Parameters:
        X_coarse - a list of arrays containing the activity of the orignal and coarse-grained variables. 
    """
    cluster_sizes = [len(c[0]) for c in clusters]
    for i, X in enumerate(X_coarse):
        # Compute correlation matrix
        corr = np.corrcoef(X)

        # Compute its eigenvalues
        eigvalues, eigvectors = np.linalg.eig(corr)

        # Plot spectrum
        sort_idx = np.argsort(eigvalues)
        eigvalues = eigvalues[sort_idx][::-1]
        rank = np.arange(0, len(eigvalues)) / len(eigvalues)
        plt.plot(rank, eigvalues, "o--", markersize=5, label=f"K = {cluster_sizes[i]}")
    
    plt.ylabel("eigenvalues")
    plt.xlabel("rank/K")
    plt.legend()
    plt.title(title)
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

def plot_eigenvalue_spectra_within_clusters(Xs, clusters, title=""):
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
        try:
            cluster_size = len(cluster[0])
        except TypeError:
            continue
        
        # Not interested in the spectra of these small clusters
        if cluster_size <= 2:
            continue
            
        # Compute the spectrum for each cluster, average and plot with std
        eigvalues_l = []
        for c in cluster:

            if len(c) != cluster_size:
                continue
            
            corr = np.corrcoef(original_dataset[c])
            eigvalues, _ = np.linalg.eig(corr)
            eigvalues_l.append(np.sort(eigvalues)[::-1])
            
        # Compute statistics
        rank = np.arange(1, len(eigvalues) + 1) / len(eigvalues)
        mean = np.mean(eigvalues_l, axis=0)
        std = np.std(eigvalues_l, axis=0)
        
        # Plot
        plt.errorbar(rank, mean, 3*std, fmt="o--", markersize=4, label=f"K = {cluster_size}")
        
    plt.xlabel("rank/K")
    plt.ylabel("eigenvalues")
    plt.legend()
    plt.title(title)
    plt.show()
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
        try:
            cluster_size = len(cluster[0])
        except TypeError:
            continue
        
        # Not interested in the spectra of these small clusters
        if cluster_size <= 2:
            continue
            
        # Compute the spectrum for each cluster, average and plot with std
        eigvalues_l = []
        for c in cluster:

            if len(c) != cluster_size:
                continue
            
            corr = np.corrcoef(original_dataset[c])
            eigvalues, _ = np.linalg.eig(corr)
            eigvalues_l.append(np.sort(eigvalues)[::-1])
            
        # Compute statistics
        rank = np.arange(1, len(eigvalues) + 1) / len(eigvalues)
        mean = np.mean(eigvalues_l, axis=0)
        std = np.std(eigvalues_l, axis=0)
        
        # Plot
        plt.errorbar(rank, mean, 3*std, fmt="o--", markersize=4, label=f"K = {cluster_size}")
        
    plt.xlabel("rank/K")
    plt.ylabel("eigenvalues")
    plt.legend()
    plt.title(title)
    plt.show()

def plot_free_energy_scaling(p_averages, p_stds, unique_activity_values, clusters, title=""):
    """
    When a RG transformation is exact the free energy does not change. This function compute the free energy at each
    coarse-grained step and log plots the values. We hope to see some scaling with a power law close to 1.

    We can compute the free energy by F = -np.ln(p0) with p0 the probability that a cluster is silent.

    Parameters:
        X_list - nd numpy array containing the variables at different steps of the coarse-graining
    """
    # Data
    p0_avg = []
    p0_std = []
    cluster_sizes = [len(c[0]) for c in clusters]
    popped = 0 # Counter to keep track of how many clusters are never silent

    # # 100% and 0% correlation limits
    # idx = np.argwhere(unique_activity_values[0] == 0.0)
    # limit100 = list(list(p_averages[0][idx])[0]) * len(unique_activity_values)
    # limit0 = (limit100) ** np.arange(1, len(unique_activity_values)+1)

    for i, unique_vals in enumerate(unique_activity_values):
        # Find idx at which cluster is silent
        idx = np.argwhere(unique_vals == 0.0)
        # Check it exists
        if len(idx) != 0:
            idx = idx[0]
            
            # Add to list
            p0_avg.append(list(p_averages[i][idx])[0])
            p0_std.append(list(p_stds[i][idx])[0])
        else:
            cluster_sizes.pop(i - popped)
            popped += 1

    # print(limit0)
            
    # # Plot limits
    # plt.plot(cluster_sizes, limit0, "--", alpha=0.5, label="0% correlation")
    # plt.plot(cluster_sizes, limit100, "--", alpha=0.5, label="100% correlation")

    # Plot the probability of the cluster being silent
    plt.errorbar(cluster_sizes, p0_avg, 3*np.array(p0_std), fmt="g^--", markersize=5)
    plt.xlabel("cluster size")
    plt.ylabel(r"P$_{Silence}$")
    plt.ylim(0, max(p0_avg)+0.1)
    #plt.yscale("log")
    plt.grid(True)
    plt.title(title)
    # plt.legend()
    plt.show()

def plot_scaling_of_variance(X_coarse, clusters, title=""):
    """
    We know that if we add to RV together their variance can be computed by Var(X+Y) = Var(X) + Var(Y) + 2Cov(X, Y). If we can assume Var(x)=Var(Y) then
    adding K uncorrelated RVs we get a scaling of the variance with K^1. On the other hand if the RVs are maximally correlated then one would expect
    a scaling with K^2 (Some assumptions were made here). 
    
    Here we plot the two limits, the scaling in the dataset and return the value a.

    Parameters:
        X_coarse - a list of size n_rg_iterations containing each a ndarray of size (n_variables, n_datapoints)
        clusters - a list of size n_rg_iterations containing the indices of the orignal spins that were clustered 

    Return:
        a - scaling found in the coarse-graining procedure
    """
    # Things to keep track of
    var_avgs = []
    var_stds = []
    cluster_sizes = []

    # Loop over RGTs
    for i, X in enumerate(X_coarse):
        cluster_size = len(clusters[0]) / len(clusters[i])
        X = X * cluster_size
        variance = np.var(X, axis=1)
        var_avgs.append(variance.mean())
        var_stds.append(variance.std())
        cluster_sizes.append(cluster_size) 
        
    # Plot K^1 limit
    limitK1 = var_avgs[0] * np.array(cluster_sizes)
    plt.plot(cluster_sizes, limitK1, "r--", alpha=0.5)
    
    # Plot K^2 limit
    limitK2 = var_avgs[0] * np.array(cluster_sizes) ** 2
    plt.plot(cluster_sizes, limitK2, "r--", alpha=0.5)
    
    # Compute log errors for plot
    var_stds = var_stds / np.abs(var_avgs)
    
    # Plot variance along with its error
    plt.errorbar(cluster_sizes, var_avgs, 3*np.array(var_stds), markersize=5, fmt="g^--")
    plt.xlabel("cluster size K")
    plt.ylabel("activity variance")
    plt.yscale("log")
    plt.xscale("log")
    plt.title(title)
    plt.show()   
