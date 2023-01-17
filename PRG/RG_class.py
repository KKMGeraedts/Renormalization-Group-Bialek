import numpy as np
from .data_cleaning import *
from . import pairwise_clustering_bialek
from . import random_pairwise_clustering
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

    def load_dataset(self, input_file, method="file"):
        """
        Load the dataset stored in location given by 'input_file'. The Dataset is then
        stored as an attribute of the class in its self.X attribute.

        Parameters:
            input_file - a string containing the relative directory of the input file

        """
        if method == "file":
            self.X = read_input(input_file)
            self.X = check_dataset(self.X.T)
        elif method == "data":
            self.X = input_file
            self.X = check_dataset(self.X.T)
        else: 
            print("Invalid method.")

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

        # Create fig, ax
        fig, ax = plt.subplots(1)

        # Plot histogram 
        ax.plot(hist_centers, hist_density, color="g")
        ax.set_xlabel("correlation coefficient")
        ax.set_ylabel("density")
        if min(hist_centers) < 0:
            ax.set_xlim(min(hist_centers), 1)
        else:
            ax.set_xlim(0, 1)
        ax.grid(True)

        # Return hist_centers and hist_value if specified
        if return_hist == True:
            return hist_centers, hist_density, fig, ax
        else:
            return fig, ax

    def _bootstrap(self, probs, N):
        """
        """
        values = []
        probs = probs.T
        for x in probs:
            values = [np.random.choice(probs[:, 0], size=len(probs[:, 0]), replace=True).mean() for i in range(N)]

    def _compute_confidence_intervals(self, df_probs, confidence=95, N=1000):
        """
        Compute confidence intervals using bootstrap

        Parameters: 
            df_probs - Dataframe containing probability of activity for each variable

        Return:
            - Confidence intervals as a 2d array.
        """
        probs = df_probs.to_numpy().T
        percentile = (100 - confidence) / 2

        # Perform a bootstrap
        confidence_intervals = np.empty(shape=(len(probs), 2))
        for i, x in enumerate(probs):
            values = [np.random.choice(x, size=len(x), replace=True).mean() for i in range(N)]
            confidence_intervals[i] = np.percentile(values, [percentile, 100-percentile])

        return confidence_intervals

    def compute_activity_distributions(self, verbose=False):
        """
        Computes the activity distributions at each intermediate step of the coarse-graining.

        # POTENTIAL UPGRADE: use scipy.stats.rv_continious or something similar to compute the pdf

        Parameters:
            verbose (optional) - if set to True prints the timing at each RG step

        # NOTE: All these below values are computed across the rg steps. Their shapes are (n_rg_iterations, ..). Where the shape in the dots is described below. 
        Return:
            p_averages - Average probability of certain activity value (across all variables). Shape(..) = unique_activity_values
            p_confidence_intervals - Confidence interval around the average probability. Shape(..) = (unique_activity_values, unique_activity_values)
            unique_activity_values - Unique activity values. Shape(..) = (unique_activity_values)
        """
        # Check that some coarse-grianing has happened
        if self.Xs == []:
            print("self.Xs == []. I.e. no coarse-graining has been performed. Try running self.perform_real_space_coarse_graining first.")
            return -1

        # Things to keep track off
        t_start_all = time.time()
        p_averages = []
        p_stds = []
        p_confidence_intervals = []
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
                
                # Add to dataframe
                new_df = pd.DataFrame.from_dict(dict(zip(values, counts)), orient="index").T
                df_probs = pd.concat([df_probs, new_df], ignore_index=True)

            # Fill nan values and sort
            df_probs = df_probs.fillna(0.00)
            df_probs = df_probs.reindex(sorted(df_probs.columns), axis=1)
                    
            # Compute column averages and stds
            p_averages.append(df_probs.mean(axis=0).to_numpy())
            p_confidence_intervals.append(self._compute_confidence_intervals(df_probs))
            p_stds.append(df_probs.std(axis=0).to_numpy())

            # Add unique values
            unique_activity = df_probs.columns.to_numpy()
            unique_activity_values.append(unique_activity)

            # Print process statistics
            if verbose:
                print(f"Finished {i+1}/{len(self.Xs)}")
                print(f"Running time = {round(time.time() - t_start, 3)} seconds.")

        print(f"Running time for activity distributions = {round(time.time() - t_start_all, 3)} seconds.")
        return p_averages, p_confidence_intervals, unique_activity_values
