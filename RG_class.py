import numpy as np
from data_cleaning import *
import pairwise_clustering_bialek
import random_pairwise_clustering
import matplotlib.pyplot as plt
import time
import pandas as pd

class RGObject():

    def __init__(self):
        self.X = []
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
            self.Xs, self.clusters, self.couplings = random_pairwise_clustering.real_space_rg(self.X, rg_iterations)
        elif method == "RBM":
            print("Not implemented error.")
        else: 
            print(f"Given method {method} does not exist. Try again by setting method to either;" \
                    "'pairwise_highest_correlated, 'random' or 'RBM'")

    def extract_data(self):
        return self.Xs, self.clusters, self.couplings

    def list_methods(self):
        print("Possible ways to perform the Renormalization group are:")
        print(" - 'random': this does a random pairwise clustering at eacht step.")
        print(" - 'pairwise_clustering_bialek': this does a pairwise clustering of the highest correlated variables.")
        print(" - 'RBM': this trains a set of Restricted Boltzmann Machines at each RG transformation.")
        return 

    def plot_correlation_structure_in_dataset(self, X=None, return_hist=False):
        """
        Plots the correlation structure in the dataset given by X. If not given then it take the original dataset 
        that is stored in self.Xs[0].

        # POTENTIAL UPGRADE: batch the dataset. Sometimes it might be to big!

        Parameters:
            X (optional) - dataset of which to study the correlation
            return_hist (optional) - returns the bin centers and bin densities

        Return (optional):
            hist_centers - the centers of the bins
            hist_density - the densities of the bins
        """
        # If no dataset was given check there is one available in self.
        if X == None:
            if self.X == []:
                print(f"No dataset was given nor was any dataset loaded, i.e. self.X==None. Try running self.load_dataset() first.")
                return -1
            else:
                X = self.X  

        # Compute correlations
        pairwise_correlations = np.corrcoef(X)

        # Create histogram values
        bins = 50
        hist_counts, hist_edges = np.histogram(pairwise_correlations, bins=bins)
        hist_centers = [(hist_edges[i] + hist_edges[i+1]) / 2 for i in range(len(hist_edges) - 1)]

        # Change histogram counts to a density
        total_count = sum(hist_counts)
        hist_density = [hist_counts[i] / total_count for i in range(len(hist_counts))]

        # Plot histogram 
        plt.plot(hist_centers, hist_density, color="g")
        plt.xlabel("correlation coefficient")
        plt.ylabel("density")
        plt.grid(True)
        plt.show()

        # Return hist_centers and hist_value if specified
        if return_hist == True:
            return hist_centers, hist_density

    def compute_probability_distributions(self, verbose=False):
        """
        Computes the probability distributions at each intermediate step of the coarse-graining.

        # POTENTIAL UPGRADE: use scipy.stats.rv_continious or something similar to compute the pdf

        Parameters:
            verbose (optional) - if set to True prints the timing at each RG step

        Return:
        probabilities_clusters - contains the probabilities in the following way, ndarray of shape (n_unique_activity_values, n_variables, n_rg_iterations) 
        activity_clusters - contains all the unique activity values, ndarray of shape (n_unique_actitiviy_values, n_rg_iterations)
        """
        # Check that some coarse-grianing has happened
        if self.Xs == []:
            print("self.Xs == []. I.e. no coarse-graining has been performed. Try running self.perform_real_space_coarse_graining first.")
            return -1

        # Things to keep track off
        t_start_all = time.time()
        p_averages = []
        p_stds = []
        unique_activity_values = []
        #df_probability_distributions = pd.DataFrame()

        # Loop over the datasets of the coarse-grained variables
        for i, X in enumerate(self.Xs):
            probabilities = []
            t_start = time.time()

            # Compute normalized activity in each cluster
            n = (2 ** (i) + 1)
            xk = np.linspace(-1, 1, n)
            pk = np.zeros(len(xk))
            average_prob = dict(zip(xk, pk))
            std_prob = dict(zip(xk, pk))
            
            df_probs = pd.DataFrame()
        
            # Loop over variables in dataset
            for var in X:
                
                # Find distribution of variable
                values, counts = np.unique(var, return_counts=True)
                counts = counts / sum(counts)
                
                df_probs = df_probs.append(dict(zip(values, counts)), ignore_index=True)

            # Compute column averages and stds
            df_probs.fillna(0.00)
            p_averages.append(df_probs.mean(axis=0).to_numpy())
            p_stds.append(df_probs.std(axis=0).to_numpy())
            
            # Add unique values
            unique_activity = df_probs.columns.to_numpy()
            unique_activity_values.append(unique_activity)
                
            # Print process statistics
            if verbose:
                print(f"Finished {i+1}/{len(self.Xs)}")
                print(f"Running time = {round(time.time() - t_start, 3)} seconds.")

        print(f"Total running time = {round(time.time() - t_start_all, 3)} seconds.")
        return p_averages, p_stds, unique_activity_values
