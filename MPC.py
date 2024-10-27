import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load the CSV file containing x and y coordinates
file_path = '/home/gopesh/Downloads/Python projects/python/pathdiv.csv'
data = pd.read_csv(file_path)
x_coords = data['70'].values
y_coords = data['20'].values

# Set parameters for the MPC
horizon = 10  # Prediction horizon
dt = 0.1      # Time step duration
num_steps = len(x_coords)  # Total number of points in path

# Initialize state (x, y position and velocities)
initial_state = [x_coords[0], y_coords[0], 0, 0]

# Define cost function for MPC
def mpc_cost(control_inputs, *args):
    x_ref, y_ref, state = args
    x, y, vx, vy = state

    cost = 0.0
    for i in range(horizon):
        # Update position based on velocities
        x += vx * dt
        y += vy * dt

        # Calculate deviation from the reference trajectory point
        dist_cost = (x - x_ref[i])**2 + (y - y_ref[i])**2
        vel_cost = vx**2 + vy**2  # Optional velocity penalization for smooth control
        cost += dist_cost + 0.1 * vel_cost  # Weighted sum of distance and velocity costs

        # Update velocity with control inputs
        vx += control_inputs[2 * i] * dt
        vy += control_inputs[2 * i + 1] * dt

    return cost

# MPC simulation
positions = [initial_state[:2]]
state = np.array(initial_state)
for step in range(0, num_steps - horizon, horizon):
    # Define the reference trajectory segment
    x_ref = x_coords[step:step + horizon]
    y_ref = y_coords[step:step + horizon]

    # Initial control inputs (0 for all)
    control_inputs = np.zeros(2 * horizon)

    # Optimize control inputs to minimize cost
    result = minimize(mpc_cost, control_inputs, args=(x_ref, y_ref, state), method='SLSQP')
    optimized_controls = result.x

    # Apply the first control inputs to the state and update position
    vx_new = state[2] + optimized_controls[0] * dt
    vy_new = state[3] + optimized_controls[1] * dt
    state = [state[0] + vx_new * dt, state[1] + vy_new * dt, vx_new, vy_new]
    positions.append(state[:2])  # Record new position

# Convert the trajectory to an array for plotting
positions = np.array(positions)

# Plot the results: reference path and MPC path
plt.figure(figsize=(10, 6))
plt.plot(x_coords, y_coords, 'r--', label='Reference Path')
plt.plot(positions[:, 0], positions[:, 1], 'b-', label='MPC Path')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.title('MPC Path Tracking')
plt.grid(True)
plt.show()
