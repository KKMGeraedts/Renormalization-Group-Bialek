import numpy as np
import matplotlib.pyplot
import PRG.RG_class as rg
import pandas as pd
from PRG.plots_to_analyse_rg_scaling import *
import copy
import time
from scipy.stats import binom
import sys, os
from PRG.data_cleaning import read_input, check_variable_type_in_dataset

# Globals
global OUTPUT_DIR

def show_data_info(RG_bialek):
    # Start
    print("\nInformation about the dataset:")
    print("=====================================")

    # Check variable type
    check_variable_type_in_dataset(RG_bialek.X)

    # Mean value and variance of two spins
    print("\n- Mean and variance of two random spins:")
    print(np.mean(RG_bialek.X[0]), np.var(RG_bialek.X[0]))
    print(np.mean(RG_bialek.X[5]), np.std(RG_bialek.X[5]))

    # Shape of the dataset
    print(f"\n- Size of dataset = {RG_bialek.X.shape}")

    # Unique values in the dataset
    print(f"\n- Unique variables in dataset: {len(np.unique(RG_bialek.X, axis=0))}")

    # Plot and save (pairwise) correlation structure in dataset
    fig, ax = RG_bialek.plot_correlation_structure_in_dataset()
    fig.savefig(f"{OUTPUT_DIR}/correlation coefficients")

    # End
    print("=====================================\n")

def perform_prg_procedure(X, n_iterations):
    # Initialize a RGObject
    RG_bialek = rg.RGObject()

    # Load dataset
    RG_bialek.load_dataset(X, method="data")

    # Show data info
    show_data_info(RG_bialek)

    # Perform Bialek RGTs
    t = time.time()
    RG_bialek.perform_real_space_coarse_graining("pairwise_clustering_bialek", n_iterations)
    print(f"Bialek RGTs toke {round(time.time() - t, 3)} seconds.")

    return RG_bialek

def create_and_save_a_series_of_plots(Xs, clusters, p_averages, p_confidence_intervals, unique_activity_values):

    # Plot scaling of variance
    moments = [2]
    fig, ax = plot_scaling_of_moments(Xs, clusters, moments=moments, limits=False)
    fig.savefig(f"{OUTPUT_DIR}/scaling_of_variance")

    # Plot scaling of a few even moments
    moments_even = np.arange(2, 13, 2)
    fig, ax = plot_scaling_of_moments(Xs, clusters, moments=moments_even, limits=False)
    fig.savefig(f"{OUTPUT_DIR}/scaling_of_even_moment2-12")

    # Plot scaling of a few odd moments
    moments_odd = np.arange(1, 13, 2)
    fig, ax = plot_scaling_of_moments(Xs, clusters, moments=moments_odd, limits=False)
    fig.savefig(f"{OUTPUT_DIR}/scaling_of_odd_moments1-11")


    # Plot free energy 
    fig, ax = plot_free_energy_scaling(p_averages, p_confidence_intervals, unique_activity_values, clusters)#TODO
    fig.savefig(f"{OUTPUT_DIR}/scaling_of_free_energy")

    # Plot eigenvalue spectra within clusters
    rg_range = (0, len(clusters)-2)
    fig, ax = plot_eigenvalue_spectra_within_clusters(Xs, clusters, rg_range)
    fig.savefig(f"{OUTPUT_DIR}/eigenvalue_spectra_within_clusters")

    # Plot distribution of activity
    rg_range = (0, len(clusters))
    fig, ax = plot_normalized_activity(p_averages, p_confidence_intervals, unique_activity_values, clusters, rg_range)#TODO
    fig.savefig(f"{OUTPUT_DIR}/normalized_activity")
 
    # Plot eigenvalue spectra between clusters
    fig, ax = plot_eigenvalue_scaling(Xs, clusters)
    fig.savefig(f"{OUTPUT_DIR}/eigenvalue_spectra_between_clusters")

    # Show the clusters at different steps by imshow
    rg_range = (0, len(clusters) - 2)
    show_clusters_by_imshow(clusters, rg_range, OUTPUT_DIR)

    print(f"Created a bunch of plots. They can be found in {OUTPUT_DIR}.")

if __name__ == "__main__":

    # Read input file from arguments, else ask user
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else: 
        print("Missing first argument, filename")
        input_file = input("Type filename here: ")

    # Read data
    X = read_input(input_file)

    # Create the output directory if it does not exist yet
    out_dir_suffix = input_file.split(".")[0][6:] # Ignore 'input/' 
    OUTPUT_DIR = f"figures/{out_dir_suffix}"
    try: 
        os.mkdir(OUTPUT_DIR)
    except FileExistsError:
        pass

    # N iterations
    if len(sys.argv) > 2:
        n_iterations = sys.argv[2]
    else:
        n_iterations = int(np.floor(np.log2(min(X.shape)) - 2))
        print(f"\nNumber of iterations was not given. Default = log2(datasize) - 2 = {n_iterations}.\n")


    # Perform PRG procedure
    RG_bialek = perform_prg_procedure(X, n_iterations)

    # Extract data
    Xs, clusters, _ = RG_bialek.extract_data()

    # Compute probability distributions (Needed for a few plots)
    p_averages, p_confidence_intervals, unique_activity_values = RG_bialek.compute_activity_distributions()

    # Create and save a series of plots
    create_and_save_a_series_of_plots(Xs, clusters, p_averages, p_confidence_intervals, unique_activity_values)

    

