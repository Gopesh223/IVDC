import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'pathdiv.csv'
data = pd.read_csv(file_path)

# Display the column names to check structure

# Extract 'x' and 'y' columns (adjust based on actual column names)
x_path = data['x'].values  # Replace 'x' with the correct column name if different
y_path = data['y'].values 
# Constants
N = 10  # Prediction horizon
dt = 0.1  # Time step
Q = np.eye(2)  # State cost matrix
R = np.eye(2)  # Control cost matrix

# Vehicle Model: x_{t+1} = f(x_t, u_t)
def vehicle_model(state, control, dt):
    x, y, theta = state
    v, omega = control
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt
    return np.array([x_next, y_next, theta_next])

# Cost function for optimization
def cost_function(variables, x0, ref_path):
    """
    Compute the cost for given states and controls over the horizon.
    variables contains both states and controls concatenated.
    """
    # Extract state and control variables from the flattened vector
    x = variables[:3 * (N + 1)].reshape((3, N + 1))
    u = variables[3 * (N + 1):].reshape((2, N))

    # Initialize the total cost
    total_cost = 0

    # Compute the cost over the prediction horizon
    for t in range(N):
        # Tracking error: (x_t - x_ref_t)^T * Q * (x_t - x_ref_t)
        state_error = x[:2, t] - ref_path[:, t]
        total_cost += state_error.T @ Q @ state_error

        # Control effort: u_t^T * R * u_t
        control_effort = u[:, t]
        total_cost += control_effort.T @ R @ control_effort

    return total_cost

# Constraints function to ensure state dynamics consistency
def dynamics_constraint(variables, x0):
    """
    Ensure that the state dynamics are respected over the horizon.
    """
    x = variables[:3 * (N + 1)].reshape((3, N + 1))
    u = variables[3 * (N + 1):].reshape((2, N))

    constraints = []
    constraints.append(x[:, 0] - x0)  # Initial state constraint

    for t in range(N):
        # Apply vehicle model dynamics
        next_state = vehicle_model(x[:, t], u[:, t], dt)
        constraints.append(x[:, t + 1] - next_state)

    # Flatten the constraints to pass to the optimizer
    return np.concatenate(constraints)

# MPC function using scipy.optimize
def mpc_control(x0, ref_path):
    # Initial guess for the optimization variables (flattened)
    x_guess = np.zeros((3, N + 1)).flatten()
    u_guess = np.zeros((2, N)).flatten()
    initial_guess = np.concatenate([x_guess, u_guess])

    # Define the optimization constraints
    constraints = {
        'type': 'eq',
        'fun': dynamics_constraint,
        'args': (x0,)  # Pass the initial state
    }

    # Solve the optimization problem
    result = minimize(
        cost_function, initial_guess, args=(x0, ref_path),
        constraints=constraints, method='SLSQP'
    )

    # Extract the optimal control inputs from the solution
    u_opt = result.x[3 * (N + 1):].reshape((2, N))
    return u_opt[:, 0]  # Return the first control input

# Simulate the vehicle following the path using MPC
x0 = np.array([x_path[0], y_path[0], 0])  # Initial state [x, y, theta]
trajectory = [x0[:2]]

for i in range(len(x_path) - N):
    # Select the reference path for the next N steps
    ref_path = np.vstack((x_path[i:i + N], y_path[i:i + N]))

    # Get optimal control inputs using MPC
    u_opt = mpc_control(x0, ref_path)

    # Apply the control input to the vehicle model
    x0 = vehicle_model(x0, u_opt, dt)

    # Store the trajectory
    trajectory.append(x0[:2])

# Convert the trajectory list to a NumPy array for plotting
trajectory = np.array(trajectory)

# Plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(x_path, y_path, 'r--', label='Reference Path')
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='MPC Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('MPC Path Following')
plt.grid(True)
plt.show()

