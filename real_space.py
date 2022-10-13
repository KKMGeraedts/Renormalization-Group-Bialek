import enum
import matplotlib.pyplot as plt 
import numpy as np
import math
from data_cleaning import *

def real_space_rg_iteration(X, correlation, test=False):
    """
    """
    # Not interested in self-correlation
    np.fill_diagonal(correlation, 0)
    X_coarse = np.zeros((X.shape[0]//2, X.shape[1]))

    if test == True:
        correlation = np.array([[0, 0, 7, 0], [0, 0, 0, 3], [7, 0, 0, 0], [0, 3, 0, 0]])
        X = np.random.randint(2, size=(4, 3))
        X_coarse = np.zeros((X.shape[0]//2, X.shape[1]))
        print("Testing the algorithm for a simple case.\n#######################################")
        print(f"The correlation matrix is given by:\n{correlation}")
        print(f"Original dataset X:\n{X}")

    # Find best pairings
    pairings = []
    list_of_original_indices = np.array([np.arange(len(X)), np.arange(len(X))])

    for i in range(len(correlation) // 2):
        # Find highest correlated pair
        max_idx = correlation.argmax()
        max_i, max_j = max_idx // len(correlation), max_idx % len(correlation)

        # Remove the corresponding row and column
        correlation = np.delete(correlation, [max_i, max_j], axis=0)
        correlation = np.delete(correlation, [max_j, max_j], axis=1)

        # Save found pairing
        max_i_original = list_of_original_indices[0][max_i]
        max_j_original = list_of_original_indices[1][max_j]
        pairings.append([max_i_original, max_j_original])
        list_of_original_indices = np.delete(list_of_original_indices, [max_i, max_j, max_i + len(list_of_original_indices[0]), max_j + len(list_of_original_indices[0])])

        # np.delete reshapes the array, undo this
        if len(list_of_original_indices) != 0:
            list_of_original_indices = list_of_original_indices.reshape(len(correlation), -1)

        # Merge pair in dataset also
        X_coarse[i] = (X[max_i_original] + X[max_j_original]) / 2

    if test == True:
        print("\nResults\n#######################################")
        print(f"Pairings found = {pairings}")
        print(f"Coarse grained dataset:\n{X_coarse}\n")

    return X_coarse, pairings

def real_space_rg(X, steps, test=False):
    """
    """
    X_list = []
    pairings_list = []
    X_coarse = X

    # Perform RG iterations
    for i in range(steps):

        # Cannot coarse any further
        if len(X_coarse == 1):
            continue

        # Compute correlation
        correlation = np.corrcoef(X_coarse)

        # Save data
        X_list.append(X_coarse)

        # RG iteration
        X_coarse, pairings = real_space_rg_iteration(X_coarse, correlation, test=test)
        pairings_list.append(pairings)

    return X_list, pairings_list

if __name__ == "__main__":
    # Read input 
    X = read_input()
    X = check_dataset_shape(X.T)

    steps = 5
    X_coarse, pairings_list = real_space_rg(X, steps, test=False)