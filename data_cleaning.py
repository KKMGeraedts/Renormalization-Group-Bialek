import numpy as np
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