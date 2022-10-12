import matplotlib.pyplot as plt 
import numpy as np
import math
from data_cleaning import *

if __name__ == "__main__":
    # Read input 
    X = read_input()
    X = check_dataset_shape(X.T)