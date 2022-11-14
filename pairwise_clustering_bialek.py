import numpy as np

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