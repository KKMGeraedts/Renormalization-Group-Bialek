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
            return np.array(X_list), np.array(clusters_list), np.array(coupling_parameters)
        elif len(X_coarse) == 2 and len(X_coarse[0]) != len(X_coarse[1]):
            return np.array(X_list), np.array(clusters_list), np.array(coupling_parameters)

        # Compute couplings
        # coupling_parameters.append(compute_couplings(X_coarse))

        # Save data
        X_list.append(X_coarse)

        # RG iteration
        X_coarse, pairings = real_space_rg_iteration(X_coarse)

        # Add pairing to clusters
        clusters = add_to_clusters(clusters, pairings)
        clusters_list.append(np.array(clusters))

    return np.array(X_list), np.array(clusters_list), np.array(coupling_parameters)

def real_space_rg_iteration(X):
    """
    Perform a single RG iteration. Given the dataset X we randomly pair two spins.

    Parameters:
        X - 2d np array containing the data

    Return:
        X_coarse - the coarse grained variables. Size = [len(X)/2, len(X)/2]
        pairings - list of indices that were paired. 
    """
    # Initialize
    X_coarse = np.zeros((X.shape[0]//2, X.shape[1]))
    X_idxs = np.arange(len(X))
    
    # Random pairings
    X_pairings_idxs = np.random.permutation(X_idxs)
        
    # Rearange the dataset to according to the permutations found
    X_rearanged = X[X_pairings_idxs]

    # Do a pairwise summation
    if (len(X_rearanged) % 2) == 0:
        X_coarse = (X_rearanged[::2] + X_rearanged[1::2]) / 2
        pairings = [[X_pairings_idxs[2*i], X_pairings_idxs[(2*i)+1]] for i in range(len(X_pairings_idxs) // 2)]
    else:
        X_coarse = (X_rearanged[1::2] + X_rearanged[2::2]) / 2
        pairings = [[X_pairings_idxs[(2*i)+1], X_pairings_idxs[(2*i)+2]] for i in range(len(X_pairings_idxs) // 2)]

        # Ignore the unpaired spin
        # X_coarse = list(X_coarse)
        # X_coarse.append(X_rearanged[0])
        # pairings.append([X_pairings_idxs[0]]

    return np.array(X_coarse), pairings

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
            #new_clusters.append(clusters[pair[0]])
            pass
        elif len(pair) == 2:
            new_cluster = np.array([clusters[pair[0]], clusters[pair[1]]])
            
            # Reshape clusters so it stays a 2d array
            new_clusters.append(new_cluster.reshape(-1))
        else:
            print("Found a pair with length > 2. Something went wrong.")
       
    return new_clusters
    