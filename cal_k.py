import numpy as np
from scipy.optimize import minimize

# Given constants based on the problem
T = 1000           
beta_1 = 0.0001    
beta_T = 0.02      
lambda_min = 0.1     # Minimum eigenvalue, gaussian distribution
R_0 = 0.5         # Initial error
w = 0.5            # Weight for inverse process error term

# Define the forward process error term based on the given parameters
def forward_process_error(k):
    integral_term = beta_1 * (T - 1) + (beta_T - beta_1) * (T - 1) / (k + 1)
    return R_0 * np.exp(0.5 * lambda_min * integral_term)

# Define the inverse process error term based on the given parameters
def inverse_process_error(k):
    integral_term = (T - 1)**2 * (beta_T - beta_1)**2 * k**2 / (2 * k + 1)
    return w * integral_term

# Objective function J(k): combination of forward and inverse errors
def objective_function(k):
    return forward_process_error(k) + inverse_process_error(k)

# Use scipy.optimize.minimize to find the optimal k
result = minimize(objective_function, x0=[1], bounds=[(0.1, 100)], method='L-BFGS-B')

# Extract optimal k value and minimum error
optimal_k = result.x[0]
min_error = result.fun  # Minimum error obtained with optimal k

print(optimal_k, min_error)
