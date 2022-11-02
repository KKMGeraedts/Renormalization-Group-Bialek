import matplotlib.pyplot as plt 
import numpy as np
import math
from data_cleaning import *

def compute_euclidean_distance(X):
    """
    TODO
    """
    x_coord = X[:, 0]
    y_coord = X[:, 1]
    
    x_coord_matrix = np.meshgrid(x_coord, x_coord)[1]
    y_coord_matrix = np.meshgrid(y_coord, y_coord)[1]
    return np.sqrt((x_coord_matrix - x_coord) ** 2 + (y_coord_matrix - y_coord) ** 2) 

def compute_covariance(X, distance_metric, pca_method):
    if distance_metric == "covariance":
        # Compute the covariance matrix or correlation matrix depending on 'pca_method' argument
        if pca_method == "covariance":
            return X @ X.T - (np.mean(X, axis=1) * np.mean(X, axis=1).T)
        elif pca_method == "correlation":
            return np.corrcoef(X)
        else:
            print(f'pca_method argument mussed be either "covariance" or "correlation. Got {pca_method} instead.')
            return None
    elif distance_metric == "euclidean":
        #TODO
        pass

def momentum_space_rg(X, step_ratio=0.05, cut_off=0.7, pca_method="correlation", distance_metric="covariance"):
    """
    Performs momentum space RG. 

    Parameters:
        X - Data array (numpy)
        cut_off - ratio of modes to be kept after last RG step.
        step_size - percentage of modes to cut at each iteration 
    
    Return:
        arr - roots
    """ 
    X_renormalized = X
    n_modes = len(X[:, 0])
    n_modes_left = n_modes

    # lists for saving some data
    u_list = []
    s_list = []
    n_modes_left_list = []
    X_list = []

    # Perform RG iterations
    while n_modes_left > n_modes * cut_off:

        # Compute covariance
        covariance = compute_covariance(X_renormalized, distance_metric, pca_method)

        # SVD
        u_original, s_original, vh = np.linalg.svd(covariance, full_matrices=False)

        # Order eigenvectors and eigenvalues from highest to lowest
        sort_idx = np.argsort(s_original)
        s_original = s_original[sort_idx][::-1]
        u_original = u_original[sort_idx][::-1]

        # Save data
        u_list.append(u_original[:n_modes_left-1])
        s_list.append(s_original[:n_modes_left-1])
        n_modes_left_list.append(n_modes_left)
        X_list.append(X_renormalized)

        # Get rid of some of the eigenmodes
        step_size = math.ceil(len(covariance) * step_ratio)
        n_modes_left -= step_size

        # Projection matrix on subset of principal components
        projection = u_original[:, :-step_size] @ u_original[:, :-step_size].T

        # Reconstruct X
        X_renormalized = projection @ X_renormalized

    return X_list, u_list, s_list, n_modes_left_list

def plot_eigenspectrum(s_list, n_modes_left_list):
    """
    Plots the eigenvalue spectrum as a function of their fractional rank for 
    several steps of the coarse-graining. 

    Parameters:
        s_list - list of eigenvalues at several steps of the coarse-graining
        n_modes_left_list - list of the number of modes left at several steps of the coarse-graining
    """
    for i, s in enumerate(s_list):
        # Plot spectrum
        fractional_rank = np.arange(0, len(s)) / len(s)
        plt.plot(fractional_rank, s, "o--", markersize=6, label=f"N = {n_modes_left_list[i]}")

    plt.xlabel("fractional rank")
    plt.ylabel("eigenvalue")
    plt.legend()
    plt.show()

def plot_normalized_fourth_moment(X_list, n_modes_left_list):
    """
    Compute and plot the normalized fourth moment of all variables across the coarse-graining.

    Parameters:
        X_list - data and reproduced data during coarse-graining
        n_modes_left_lost - number of modes left at acroos the coarse-graining
    """
    for i, X in enumerate(X_list):
        # compute fourth moment
        normalized_fourth_moment = np.mean(X**4, axis=1) / (np.mean(X**2, axis=1) ** 2)

        # plot
        fraction_of_modes_left = round(n_modes_left_list[i] / n_modes_left_list[0], 2)
        plt.errorbar(
            fraction_of_modes_left, np.mean(normalized_fourth_moment),
            yerr=3*np.std(normalized_fourth_moment),
            marker="o",
            markersize=6
            )

    # Plot Gaussian fixed point
    print(np.array(n_modes_left_list).shape, np.array(X_list).shape)
    fraction_of_modes_left_list = np.array(n_modes_left_list) / n_modes_left_list[0]
    plt.plot(fraction_of_modes_left_list, np.zeros(len(X_list)) + 3, "--", label="Gaussian fixed point")

    plt.ylabel("Normalized fourth moment")
    plt.xlabel("Fraction of modes left")
    plt.legend()
    plt.show()

def plot_eigenvectors(u_list, n):
    """

    Parameters:
        u_list - list of eigenvectors during coarse-graining
        n - number of largest eigenvalues to plot
    """
    for i in range(n):

        fig, ax = plt.subplots(1, 2)    

        # Change indecis i (i think)
        ax[0].imshow(u_list[0][:, i].reshape(u_list[0][:, i].size, 1) @ u_list[0][i].reshape(u_list[0][i].size, 1).T)
        ax[0].set_title(f"Before coarse-graining")

        ax[1].imshow(u_list[-1][i].reshape(u_list[0][i].size, 1) @ u_list[-1][i].reshape(u_list[0][i].size, 1).T)
        ax[1].set_title(f"After {len(u_list)} steps of coarse-grainig")
    
        plt.show()

if __name__ == "__main__":

    # Read input 
    X = read_input()
    X = check_dataset_shape(X.T)
    print(f"Shape of input data = {X.shape}")

    # Perform RG in momentum space 
    cut_off = 0.5
    step_ratio = 0.10
    X_list, u_list, s_list, n_modes_left_list = momentum_space_rg(X, step_ratio, cut_off, pca_method="covariance")

    # Plot eigenspectrum
    plot_eigenspectrum(s_list, n_modes_left_list)

    # Plot normalized fourth moment
    plot_normalized_fourth_moment(X_list, n_modes_left_list)

    # Show n largest eigenvectors 
    n = len(u_list[-1])
    print(n)
    plot_eigenvectors(u_list, n)