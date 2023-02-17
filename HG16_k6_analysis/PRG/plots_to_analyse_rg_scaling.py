import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
from scipy.stats import binom, moment
import matplotlib.colors as mcolors

def plot_normalized_activity(p_averages, p_confidence_intervals, unique_activity_values, clusters, rg_range=(0,0), title=""):
    """
    Plots the distribution of the normalized activity. Given the average probabilities, standard deviations and the unique values at each
    step of the coarse-graining. This data can be obtained from the RG_class.

    Parameters:
        p_averages - a list of size = n_rg_iterations with each list containing an numpy array of the average activity across all clusters
        p_confidence_intervals - a list of size = n_rg_iterations with each item containing a list of upper and lower values for 95% confidence interval
        unique_activity_values - a list of size = n_rg_iterations with each list containing an numpy array of the unique activity values in the clusters
    """
    cluster_sizes = [len(clusters[0]) / len(c) for c in clusters]

    # Create fig, ax
    fig, ax = plt.subplots(1)
    
    if rg_range != (0,0):
        cluster_sizes = cluster_sizes[rg_range[0]:rg_range[1]]
        p_averages = p_averages[rg_range[0]:rg_range[1]]
        p_confidence_intervals = p_confidence_intervals[rg_range[0]:rg_range[1]]
        unique_activity_values = unique_activity_values[rg_range[0]:rg_range[1]]
        clusters = clusters[rg_range[0]:rg_range[1]]

    for i, _ in enumerate(p_averages):
        p_confidence_interval = np.abs(p_confidence_intervals[i].T - p_averages[i])
        ax.errorbar(unique_activity_values[i], p_averages[i], yerr=np.array(p_confidence_interval), fmt="o--", markersize=3, label=f"K = {cluster_sizes[i]}")

        # Plot probability distribution of binomial p=0.5 with n=cluster_size
        n = cluster_sizes[i]
        p = 0.5
        x = np.arange(0, n+1)
        ax.plot(x / cluster_sizes[i], binom.pmf(x, n, p), '--', color="grey", alpha=0.3)
        
    # Make plot nice
    ax.set_ylabel("probability")
    ax.set_xlabel("normalized activity")
    ax.set_title("Probability distribution of the normalized activity")
    ax.grid(True)
    ax.legend(loc="upper right")

    return fig, ax

def show_clusters_by_imshow(clusters, rg_range, OUTPUT_DIR):
    """
    Show the clusters that are formed during the RG procedure.

    Parameters:
        cluster - ndarray containing the clusters at the different iterations of the RG transformation
        verbose (optional) - if True it prints the cluster list after showing the image 
    """

    original_size = len(clusters[0][0]) * len(clusters[0][:, 0])
    grid = np.zeros((original_size, original_size))
    for c in clusters[rg_range[0]:rg_range[1]]:

        # Create fig, ax
        fig, ax = plt.subplots(1)

        colors = np.arange(1, 1 + len(c[:,0])) / len(c[:,0]) * 10
        for i, color in enumerate(colors):
            for j in c[i]:
                grid[j, c[i]] = color
        
        ax.imshow(grid)
        ax.set_title(f"cluster size = {len(c.T)}")

        if OUTPUT_DIR != "":
            fig.savefig(f"{OUTPUT_DIR}/clusterSize={len(c.T)}")

def plot_eigenvalue_scaling(X_coarse, clusters, rg_range=(0,0)):
    """
    Plot the eigenvalues spectrum of the correlation function at different steps of the 
    coarse graining.

    Parameters:
        X_coarse - a list of arrays containing the activity of the orignal and coarse-grained variables. 
    """
    # Create fig, ax
    fig, ax = plt.subplots(1)

    if rg_range != (0,0):
        X_coarse = X_coarse[rg_range[0]:rg_range[1]]
        clusters = clusters[rg_range[0]:rg_range[1]]

    cluster_sizes = [len(c[0]) for c in clusters]
    for i, X in enumerate(X_coarse):
        # Compute correlation matrix
        corr = np.corrcoef(X)

        # Compute its eigenvalues
        eigvalues, eigvectors = np.linalg.eig(corr)

        # Check complex part of eigenvalues
        delta = 10e-3
        large_complex_eigvalues = eigvalues.imag[eigvalues.imag > delta]
        if large_complex_eigvalues != []:
            print(f"Found some eigenvalues with complex part larger than {delta}. Ignoring them for now. {large_complex_eigvalues}")

        eigvalues = eigvalues.real

        # Plot spectrum
        sort_idx = np.argsort(eigvalues)
        eigvalues = eigvalues[sort_idx][::-1]
        rank = np.arange(0, len(eigvalues)) / len(eigvalues)
        ax.plot(rank, eigvalues, "o--", markersize=5, label=f"K = {cluster_sizes[i]}")

    # Make plot nice
    ax.set_ylabel("eigenvalues")
    ax.set_xlabel("rank/K")
    ax.set_yscale("log")
    ax.legend()

    return fig, ax

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

def plot_eigenvalue_spectra_within_clusters(Xs, clusters, rg_range=(0,0)):
    """
    This function plots the eigenvalue spectra within the clusters. At each coarse-grained level the mean and variance of the spectra
    across the different clusters are computed and plotted.

    Parameters:
        Xs - list contianing the dataset at each coarse-grained level
        clusters - list containing the clusters that where formed at the different coarse-grianing iterations
    """
    original_dataset = Xs[0]

    if rg_range != (0,0):
        Xs = Xs[rg_range[0]:rg_range[1]]
        clusters = clusters[rg_range[0]:rg_range[1]]

    # Create figure and ax
    fig, ax = plt.subplots(1)

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
            
        # Compute the spectrum for each cluster, average and plot with confidence interval
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

        # Bootstrap params
        N = 1000
        percentile = 2.5 # =(100-confidence)/2 
        confidence_intervals = np.empty(shape=(len(eigvalues_l[0]), 2))

        # Perform bootstrap for the confidence interval
        for j, eigvs in enumerate(np.transpose(eigvalues_l)):
            bootstrap_values = [np.random.choice(eigvs, size=len(eigvs), replace=True).mean() for i in range(N)]
            confidence_intervals[j] = np.percentile(bootstrap_values, [percentile, 100-percentile])
            confidence_intervals[j] = np.abs(confidence_intervals[j] - mean[j])
    
        # if cluster_size == 32:
        #     print(np.array(eigvalues_l)[:, 1])
        #     print(np.mean(np.array(eigvalues_l)[:, 1]))

        # Plot
        ax.errorbar(rank, mean, yerr=confidence_intervals.T, fmt="o--", markersize=4, label=f"K = {cluster_size}")

    ax.set_xlabel("rank/K")
    ax.set_ylabel("eigenvalues")
    ax.set_yscale("log")
    ax.legend()

    return fig, ax

def plot_free_energy_scaling(p_averages, p_confidence_intervals, unique_activity_values, clusters):
    """
    When a RG transformation is exact the free energy does not change. This function compute the free energy at each
    coarse-grained step and log plots the values. We hope to see some scaling with a power law close to 1.

    We can compute the free energy by F = -np.ln(p0) with p0 the probability that a cluster is silent.

    Parameters:
        X_list - nd numpy array containing the variables at different steps of the coarse-graining
    """
    # Data
    p0_avg = []
    p0_confidence_intervals = []
    cluster_sizes = [len(c[0]) for c in clusters]
    popped = 0 # Counter to keep track of how many clusters are never silent

    # # 100% and 0% correlation limits
    # idx = np.argwhere(unique_activity_values[0] == 0.0)
    # limit100 = list(list(p_averages[0][idx])[0]) * len(unique_activity_values)
    # limit0 = (limit100) ** np.arange(1, len(unique_activity_values)+1)

    # Create fig, ax
    fig, ax = plt.subplots(1)

    for i, unique_vals in enumerate(unique_activity_values):
        # Find idx at which cluster is silent
        idx = np.argwhere(unique_vals == 0.0)
        # Check it exists
        if len(idx) != 0:
            idx = idx[0]
            
            # Add to list
            p0_avg.append(list(p_averages[i][idx])[0])
            p0_confidence_intervals.append([p_confidence_intervals[i][idx][0][0], p_confidence_intervals[i][idx][0][1]])
        else:
            cluster_sizes.pop(i - popped)
            popped += 1

    # print(limit0)
            
    # # Plot limits
    # plt.plot(cluster_sizes, limit0, "--", alpha=0.5, label="0% correlation")
    # plt.plot(cluster_sizes, limit100, "--", alpha=0.5, label="100% correlation")

    # Plot the probability of the cluster being silent
    p0_confidence_intervals = np.abs(np.transpose(p0_confidence_intervals) - p0_avg)  
    ax.errorbar(cluster_sizes, p0_avg, yerr=p0_confidence_intervals, fmt="g^--", markersize=5)
    ax.set_xlabel("cluster size")
    ax.set_ylabel(r"P$_{Silence}$")
    ax.set_ylim(0, max(p0_avg)+0.1)
    #plt.yscale("log")
    ax.grid(True)

    return fig, ax

def plot_scaling_of_moments(X_coarse, clusters, moments=[2], limits=True):
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
    fig, ax = plt.subplots(1)
    x = []
    y = []
    yerr = []
    for n_th_moment in moments[::-1]:

        # Things to keep track of
        moment_avgs = []
        confidence_intervals = np.empty(shape=(len(X_coarse), 2))
        cluster_sizes = []

        # Loop over RGTs
        for i, X in enumerate(X_coarse):

            # Compute cluster size and save
            cluster_size = len(clusters[0]) / len(clusters[i])
            cluster_sizes.append(cluster_size)

            X = X * cluster_size # Unnormalize the activity

            # Compute moment
            n_moment = moment(X, moment=n_th_moment, axis=1) # These are the central moments

            # Compute mean
            moment_avgs.append(n_moment.mean())

            # Compute confidence interval by bootstrap
            N = 1000
            percentile = 2.5 
            bootstrap_values = [np.random.choice(n_moment, size=(len(n_moment))).mean() for _ in range(N)]
            confidence_interval = [np.percentile(bootstrap_values, percentile), np.percentile(bootstrap_values, 100-percentile)]
            confidence_intervals[i] = np.abs(moment_avgs[i] - confidence_interval)

            # if i == 0:
            #     y.append(moment_avgs[0])
            #     yerr.append(3*np.array(moment_stds[0]))
        
        # # Compute log errors for plot
        # with np.errstate(invalid='ignore'):
        #     moment_stds = moment_stds / np.abs(moment_avgs)

        # Plot moments along with error
        #ax.plot(cluster_sizes, moment_avgs, "^", label=f"n = {n_th_moment}")
        ax.errorbar(cluster_sizes, moment_avgs, confidence_intervals.T, markersize=5, fmt="^--", label=f"n = {n_th_moment}")
        
        if limits == True:
            # # Plot K^1 limit (for variance)
            limitK1 = moment_avgs[0] * np.array(cluster_sizes)
            ax.plot(cluster_sizes, limitK1, "r--", alpha=0.5)

            # # Plot K^2 limit (for variance)
            limitK2 = moment_avgs[0] * np.array(cluster_sizes) ** n_th_moment
            ax.plot(cluster_sizes, limitK2, "r--", alpha=0.5)

    # Make figure look nice
    ax.set_xlabel("cluster size K")
    ax.set_ylabel("activity variance")
    ax.set_yscale("log")
    ax.set_xscale("log")

    if len(moments) > 1:
        ax.legend()

    # plt.scatter(moments[::-1], y)
    # plt.title("Central Moments")
    # plt.plot

    return fig, ax
