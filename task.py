import numpy as np
import matplotlib.pyplot as plt

# Given data
x = np.array([5, 10, 20, 30])
y = np.array([10, 15, 25, 35])
theta = np.array([1, 1])  # initial parameters
alpha = 0.002  # learning rate

# Perform gradient descent
num_iterations = 100
m = len(y)

# Lists to store the cost values and iteration numbers
cost_values = []
iterations_list = []

for iteration in range(num_iterations):
    h = theta[0] + theta[1] * x
    error = h - y

    # Initialize gradients and cost
    dJ_dθ0_sum = 0
    dJ_dθ1_sum = 0
    cost_sum = 0

    # Compute gradients and accumulate cost
    for i in range(m):
        dJ_dθ0 = error[i]
        dJ_dθ1 = error[i] * x[i]
        cost = (error[i] ** 2)
        dJ_dθ0_sum += dJ_dθ0
        dJ_dθ1_sum += dJ_dθ1
        cost_sum += cost

    # Update parameters
    new_theta0 = theta[0] - alpha * dJ_dθ0_sum / m
    new_theta1 = theta[1] - alpha * dJ_dθ1_sum / m
    theta = np.array([new_theta0, new_theta1])

    # Store cost value and iteration number
    cost_values.append(cost_sum / (2 * m))
    iterations_list.append(iteration + 1)

# Plot the cost values over iterations
plt.plot(iterations_list, cost_values, marker='o')
plt.title('Cost Values over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.grid(True)
plt.show()
