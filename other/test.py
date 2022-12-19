import numpy as np

def find_optimal_parameters(model_operators, df_data, n_spins, epsilon):
    N = 895
    df_data[[0,1,2,3,4,5,6,7,8]] = df_data[[0,1,2,3,4,5,6,7,8]].astype(float)
    # Initial parameters
    initial_params = np.random.rand(len(model_operators))
    # Compute operator averages in data
    data_operator_averages = np.zeros(len(model_operators))
    for i in range(len(model_operators)):
        data_operator_averages[i] = calc_average_of_operator(df_data, model_operators[i])
        
    params_optimal = minimize(neg_log_likelihood, initial_params, args=(model_operators, data_operator_averages, N, n_spins), method='L-BFGS-B', tol=epsilon)
    return params_optimal