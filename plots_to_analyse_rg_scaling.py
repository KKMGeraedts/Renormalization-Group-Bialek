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

def plot_probability_distributions(probabilities_clusters, activity_clusters):

    probs = np.array(probabilities_clusters)
    activs = [activ[0] for activ in activity_clusters]

    # free energy
    free_energy = []
    free_energy_err = []
    cluster_sizes = []
    search_value = activs[1][1]

    fig, ax = plt.subplots(1)

    for i, prob in enumerate(probs):
        mean_probs = np.mean(prob, axis=0)
        std_probs = np.std(prob, axis=0)
        
        cluster_sizes.append(len(activity_clusters[0]) // len(activity_clusters[i]))
        plt.errorbar(activs[i], mean_probs, 2*std_probs, fmt="o--", markersize=3, label=f"K = {cluster_sizes[i]}")

        if i != 0:
            p0 = mean_probs[np.where(activs[i] == search_value)]
            p0_std = std_probs[np.where(activs[i] == search_value)]
            free_energy.append(np.log(p0)[0])
            free_energy_err.append(-np.log(p0_std)[0])

    plt.ylabel("probability")
    plt.xlabel("normalized activity")
    plt.grid(True)
    plt.legend()
    plt.show()

    #plt.errorbar(cluster_sizes[1:], free_energy, free_energy_err, fmt="go--", markersize=5)
    plt.plot(cluster_sizes[1:], free_energy, "go--", markersize=5)
    plt.xlabel("cluster size")
    plt.ylabel(r"ln(P$_{silence}$)")
    #plt.yscale("log")
    plt.grid(True)
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
        
    plt.xlabel("rank / K")
    plt.ylabel("eigenvalues")
    plt.legend()
    plt.show()

def plot_free_energy_scaling(X_list):
    """
    When a RG transformation is exact the free energy does not change. This function compute the free energy at each
    coarse-grained step and log plots the values. We hope to see some scaling with a power law close to 1.

    Parameters:
        X_list - nd numpy array containing the variables at different steps of the coarse-graining
    """
    
    # Loop over the datasets of the coarse-grained variables
    for X in X_list:
        pass