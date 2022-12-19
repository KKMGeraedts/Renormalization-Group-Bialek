import numpy as np
import sys

def read_input(input_file=None):
    """
    Reads filename from arguments else asks user for input if no filename was given.
    So far only works with numpy specific file formats: .npy and .dat.

    Return:
        Numpy array containing data.
    """
    # Read filename from arguments or ask if none were given
    # if len(sys.argv) == 1:
    #     input_file = input("Input filename = ")
    if input_file == None:
        input_file = sys.argv[1]

    # Check file type, read file and return np array
    if input_file[-3:] == "npy":
        return np.load(f"./{input_file}")
    elif input_file[-3:] == "dat":
        # Assuming .dat files contain at each row a binary string, eg. '00..01100'
        return np.genfromtxt(f"./{input_file}", delimiter=1, dtype=np.int8)  
    else:
        print("Make sure input file has extensions .npy or .dat")
        return None

def check_dataset(X):
    X = check_dataset_shape(X)
    check_variable_type_in_dataset(X)
    return X

def check_variable_type_in_dataset(X):
    """
    This function checks whether the dataset contains binary variables and what these binary values are.
    """
    X_unique = np.unique(X)

    if len(X_unique) == 2:
        print(f"Dataset contains binary values [{X_unique[0]}, {X_unique[1]}].")
    elif len(X_unique) == 3:
        print(f"Dataset contains triple values [{X_unique[0]}, {X_unique[1]}, {X_unique[2]}]")
    else:
        print(f"Dataset does not contain binary values. Found {len(X_unique)} unique values.")

def check_dataset_shape(X):
    """
    Computation of the correlation matrix assumes the shape = (n_features, n_datapoints). 
    Perform a simple check and ask user if they want to transpose the data.
    """
    if len(X[:, 0]) > len(X[0]):
        response = input(f"Dataset has shape = {X.shape}. There are more features than data points! Do you want to transpose the data? ")
        if response in ["yes", "y", "YES", "ye", "yh", "Yes"]:
            return X.T
    return X