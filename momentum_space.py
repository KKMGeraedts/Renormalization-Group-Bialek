import enum
from pickle import NONE
from socketserver import ForkingUDPServer
import matplotlib.pyplot as plt 
import numpy as np
import math
import sys

def read_input():
    """
    Asks user for input file.
    So far only works with numpy specific file formats: .npy and .dat.

    Return:
        Numpy array containing data.
    """
    # Read filename from arguments or ask if none were given
    if len(sys.argv) == 1:
        input_file = input("Input filename = ")
    else:
        input_file = sys.argv[1]

    # Check file type, read file and return np array
    if input_file[-3:] == "npy":
        return np.load(f"./input/{input_file}")
    elif input_file[-3:] == "dat":
        return np.loadtxt(f"./input/{input_file}")
    else:
        print("Make sure input file has extensions .npy or .dat")
        return None

def check_dataset_shape(X):
    """
    Computation of the correlation matrix assumes the shape = (n_features, n_datapoints). 
    Perform a simple check and ask user if they want to transpose the data.
    """
    if len(X[:, 0]) > len(X[0]):
        response = input(f"Dataset has shape = {X.shape}.\nThere are more features than data points! Shall I transpose the data? ")
        if response in ["yes", "y", "YES", "ye", "yh", "Yes"]:
            return X.T
    return X

def momentum_space_rg(X, step_ratio=0.05, cut_off=0.7, type="covariance"):
    """
    Performs momentum space RG. 

    Input:
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

    # Compute the covariance matrix or correlation matrix depending on 'type' argument
    if type == "covariance":
        covariance = X @ X.T - (np.mean(X, axis=1) * np.mean(X, axis=1).T)
    elif type == "correlation":
        covariance = np.corrcoef(X)
    else:
        print(f'Type argument mussed be either "covariance" or "correlation. Got {type} instead.')
        return None

    # eigv, eigh = np.linalg.eig(covariance)
    # eigv_sorted = np.sort(eigv)[::-1]

    # plt.plot(range(len(eigv_sorted)), eigv_sorted)
    # plt.title("covariance")
    # plt.show()

    # SVD
    u_original, s_origianl, vh = np.linalg.svd(covariance, full_matrices=False)

    # Perform RG iterations
    while n_modes_left > n_modes * cut_off:

        # Compute the covariance matrix
        covariance = X_renormalized @ X_renormalized.T - (np.mean(X, axis=1) * np.mean(X, axis=1).T)

        # SVD
        u, s, vh = np.linalg.svd(covariance, full_matrices=False)

        # Order eigenvectors and eigenvalues from highest to lowest
        sort_idx = np.argsort(s)[::-1]
        s = s[sort_idx]
        u = u[sort_idx]

        # Save data
        u_list.append(u_original[:n_modes_left-1])
        s_list.append(s_origianl[:n_modes_left-1])
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

    Input:
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

    Input:
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
    fraction_of_modes_left_list = np.array(n_modes_left_list) / n_modes_left_list[0]
    plt.plot(fraction_of_modes_left_list, np.zeros(len(X_list)) + 3, "--", label="Gaussian fixed point")

    plt.ylim(0, 5)
    plt.ylabel("Normalized fourth moment")
    plt.xlabel("Fraction of modes left")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":

    # Read input 
    X = read_input()
    print(type(X))
    X = check_dataset_shape(X.T)

    # Perform RG in momentum space 
    X_list, u_list, s_list, n_modes_left_list = momentum_space_rg(X, cut_off=0.7)

    # Plot eigenspectrum
    plot_eigenspectrum(s_list, n_modes_left_list)

    # Plot normalized fourth moment
    plot_normalized_fourth_moment(X_list, n_modes_left_list)