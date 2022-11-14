import numpy as np
from data_cleaning import *
import pairwise_clustering_bialek

class RGObject():

    def __init__(self):
        self.X = None
        self.N_spins = None
        self.Xs = []
        self.clusters = []
        self.couplings = []

    def load_dataset(self, input_file):
        """
        Load the dataset stored in location given by 'input_file'. The Dataset is then
        stored as an attribute of the class in its self.X attribute.

        Parameters:
            input_file - a string containing the relative directory of the input file

        """
        self.X = read_input(input_file)
        self.X = check_dataset(self.X.T)

    def perform_real_space_coarse_graining(self, method, rg_iterations=5):
        """
        Performs a real space rg procedure given a chosen method. The possible methods are
        1) pairwise_clustering_bialek
        2) random
        3) RBM (#TODO)

        Parameters:
            method - string containing the rg procedure the user wants to use.
            rg_iterations (optional) - integer containing the number of rg iterations to perform
        """

        #np.array(X_list), np.array(clusters_list), np.array(coupling_parameters)

        if method == "pairwise_clustering_bialek":
            self.Xs, self.clusters, self.couplings = pairwise_clustering_bialek.real_space_rg(self.X, rg_iterations)
        elif method == "random":
            pass
        elif method == "RBM":
            print("Not implemented error.")
        else: 
            print(f"Given method {method} does not exist. Try again by setting method to either;" \
                    "'pairwise_highest_correlated, or 'random' or 'RBM'")

    def extract_data(self):
        return self.Xs, self.clusters, self.couplings
        