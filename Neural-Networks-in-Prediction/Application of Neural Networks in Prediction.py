#!/usr/bin/env python
# coding: utf-8

# # EE 399 Introduction to Machine Learning: HW 5 Submission
# ### Sabrina Hwang

# In[2]:


# GitHub HW5: https://github.com/hwangsab/EE-399-HW5


# ### Problem 1
# Train a NN to advance the solution from t to t + ∆t for ρ = 10, 28 and 40. Now see how well your NN works for future state prediction for ρ = 17 and ρ = 35.

# In[11]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the Lorenz equations
def lorenz(state, t, rho, sigma, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Time steps
t_start = 0.0
t_end = 10.0
delta_t = 0.01
num_steps = int((t_end - t_start) / delta_t) + 1

# Parameters
rho_values = [10, 28, 40]
sigma = 10.0
beta = 8.0 / 3.0

# Solve the Lorenz equations for each rho value
for rho in rho_values:
    # Generate initial conditions
    initial_state = np.random.randn(3)
    
    # Solve the equations using odeint
    t = np.linspace(t_start, t_end, num_steps)
    states = odeint(lorenz, initial_state, t, args=(rho, sigma, beta))
    
    # Prepare data for training the neural network
    inputs = states[:-1]
    targets = states[1:]
    
    # Create and train the neural network
    model = Sequential()
    model.add(Dense(64, input_shape=(3,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mse')
    model.fit(inputs, targets, epochs=10, batch_size=32)
    
    # Generate predictions using the neural network
    predicted_states = model.predict(inputs)
    
    # Compute mean square error (MSE)
    mse = mean_squared_error(targets, predicted_states)
    print(f"MSE (ρ = {rho}): {mse:.6f}")
    
    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], label='Actual')
    ax.plot(predicted_states[:, 0], predicted_states[:, 1], predicted_states[:, 2], label='Predicted')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Lorenz Equations (ρ = {rho})')
    ax.legend()
    plt.show()


# In[12]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the Lorenz equations
def lorenz(state, t, rho, sigma, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Time steps
t_start = 0.0
t_end = 10.0
delta_t = 0.01
num_steps = int((t_end - t_start) / delta_t) + 1

# Parameters
rho_values = [17, 35]
sigma = 10.0
beta = 8.0 / 3.0

# Solve the Lorenz equations for each rho value and obtain the initial condition
initial_state = np.random.randn(3)
predicted_states = []

# Create and train the neural network
model = Sequential()
model.add(Dense(64, input_shape=(3,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')

for rho in rho_values:
    # Solve the equations using odeint for the given rho value
    t = np.linspace(t_start, t_end, num_steps)
    states = odeint(lorenz, initial_state, t, args=(rho, sigma, beta))
    
    # Prepare data for training the neural network
    inputs = states[:-1]
    targets = states[1:]
    
    # Train the neural network
    model.fit(inputs, targets, epochs=10, batch_size=32)
    
    # Store the predicted states
    predicted_states.append(states)
    
    # Update the initial condition for the next rho value
    initial_state = states[-1]

# Perform future state predictions
future_time = np.linspace(t_end, t_end + 10, num_steps)
future_states = []

for rho, initial_state in zip(rho_values, predicted_states):
    # Generate inputs for future state predictions
    inputs = initial_state[-1:]
    
    # Predict future states using the trained model
    predicted = model.predict(inputs)
    future_states.append(np.concatenate((initial_state, predicted)))
    
    # Update the initial condition for the next rho value
    initial_state = np.concatenate((initial_state, predicted))[-num_steps:]

# Plot the results
for rho, states in zip(rho_values, future_states):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Lorenz Equations Future Predictions (ρ = {rho})')
    plt.show()


# ### Problem 2
# Compare feed-forward, LSTM, RNN and Echo State Networks for forecasting the dynamics.

# In[2]:


# Set the values of ρ
rhos = [10, 28, 40]

# Define the Lorenz system equations
def lorenz_equations(state, rho):
    x, y, z = state
    dx_dt = rho * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - (rho / 8) * z
    return [dx_dt, dy_dt, dz_dt]

# Generate training data
def generate_data(rho, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    n_steps = len(t)
    state = [1.0, 1.0, 1.0]  # Initial state
    data = np.zeros((n_steps, 3))
    for i in range(n_steps):
        data[i] = state
        state_dt = lorenz_equations(state, rho)
        state = [state[j] + state_dt[j] * dt for j in range(3)]
    return data


# Feed-forward network:

# In[5]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the Lorenz equations
def lorenz(state, t, rho, sigma, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Define the feed-forward neural network
def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3))
    return model

# Time steps
t_start = 0.0
t_end = 10.0
delta_t = 0.01
num_steps = int((t_end - t_start) / delta_t) + 1

# Parameters
rho_values = [10, 28, 40]
sigma = 10.0
beta = 8.0 / 3.0

# Solve the Lorenz equations for each rho value
for rho in rho_values:
    # Generate initial conditions
    initial_state = np.random.randn(3)
    
    # Solve the equations using odeint
    t = np.linspace(t_start, t_end, num_steps)
    states = odeint(lorenz, initial_state, t, args=(rho, sigma, beta))
    
    # Prepare data for training the neural network
    inputs = states[:-1]
    targets = states[1:]
    
    # Create and train the neural network
    model = create_model()
    model.compile(optimizer='adam', loss='mse')
    model.fit(inputs, targets, epochs=10, batch_size=32)
    
    # Generate predictions using the neural network
    predicted_states = model.predict(inputs)
    
    # Compute mean square error (MSE)
    mse = mean_squared_error(targets, predicted_states)
    print(f"MSE (ρ = {rho}): {mse:.6f}")
    
    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], label='Actual')
    ax.plot(predicted_states[:, 0], predicted_states[:, 1], predicted_states[:, 2], label='Predicted')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Lorenz Equations (ρ = {rho})')
    ax.legend()
    plt.show()


# LSTM Network:

# In[6]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the Lorenz equations
def lorenz(state, t, rho, sigma, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Time steps
t_start = 0.0
t_end = 10.0
delta_t = 0.01
num_steps = int((t_end - t_start) / delta_t) + 1

# Parameters
rho_values = [10, 28, 40]
sigma = 10.0
beta = 8.0 / 3.0

# Solve the Lorenz equations for each rho value
for rho in rho_values:
    # Generate initial conditions
    initial_state = np.random.randn(3)
    
    # Solve the equations using odeint
    t = np.linspace(t_start, t_end, num_steps)
    states = odeint(lorenz, initial_state, t, args=(rho, sigma, beta))
    
    # Prepare data for training the LSTM network
    inputs = states[:-1]
    targets = states[1:]
    
    # Reshape inputs for LSTM (samples, time steps, features)
    inputs = np.reshape(inputs, (inputs.shape[0], 1, inputs.shape[1]))
    
    # Create and train the LSTM network
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, 3), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mse')
    model.fit(inputs, targets, epochs=10, batch_size=32)
    
    # Generate predictions using the LSTM network
    predicted_states = model.predict(inputs)
    
    # Compute mean square error (MSE)
    mse = mean_squared_error(targets, predicted_states)
    print(f"MSE (ρ = {rho}): {mse:.6f}")
    
    # Reshape predicted states for plotting
    predicted_states = np.reshape(predicted_states, (predicted_states.shape[0], predicted_states.shape[1]))
    
    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], label='Actual')
    ax.plot(predicted_states[:, 0], predicted_states[:, 1], predicted_states[:, 2], label='Predicted')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Lorenz Equations (ρ = {rho})')
    ax.legend()
    plt.show()


# RNN Network:

# In[7]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Define the Lorenz equations
def lorenz(state, t, rho, sigma, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Time steps
t_start = 0.0
t_end = 10.0
delta_t = 0.01
num_steps = int((t_end - t_start) / delta_t) + 1

# Parameters
rho_values = [10, 28, 40]
sigma = 10.0
beta = 8.0 / 3.0

# Solve the Lorenz equations for each rho value
for rho in rho_values:
    # Generate initial conditions
    initial_state = np.random.randn(3)
    
    # Solve the equations using odeint
    t = np.linspace(t_start, t_end, num_steps)
    states = odeint(lorenz, initial_state, t, args=(rho, sigma, beta))
    
    # Prepare data for training the RNN network
    inputs = states[:-1]
    targets = states[1:]
    
    # Reshape inputs for RNN (samples, time steps, features)
    inputs = np.reshape(inputs, (inputs.shape[0], 1, inputs.shape[1]))
    
    # Create and train the RNN network
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(1, 3), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mse')
    model.fit(inputs, targets, epochs=10, batch_size=32)
    
    # Generate predictions using the RNN network
    predicted_states = model.predict(inputs)
    
    # Compute mean square error (MSE)
    mse = mean_squared_error(targets, predicted_states)
    print(f"MSE (ρ = {rho}): {mse:.6f}")
    
    # Reshape predicted states for plotting
    predicted_states = np.reshape(predicted_states, (predicted_states.shape[0], predicted_states.shape[1]))
    
    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], label='Actual')
    ax.plot(predicted_states[:, 0], predicted_states[:, 1], predicted_states[:, 2], label='Predicted')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Lorenz Equations (ρ = {rho})')
    ax.legend()
    plt.show()


# Echo State Networks:

# In[10]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from scipy.sparse import random as sparse_random

# Define the Lorenz equations
def lorenz(state, t, rho, sigma, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Time steps
t_start = 0.0
t_end = 10.0
delta_t = 0.01
num_steps = int((t_end - t_start) / delta_t) + 1

# Parameters
rho_values = [10, 28, 40]
sigma = 10.0
beta = 8.0 / 3.0

# Solve the Lorenz equations for each rho value
for rho in rho_values:
    # Generate initial conditions
    initial_state = np.random.randn(3)
    
    # Solve the equations using odeint
    t = np.linspace(t_start, t_end, num_steps)
    states = odeint(lorenz, initial_state, t, args=(rho, sigma, beta))
    
    # Prepare data for training the ESN
    inputs = states[:-1]
    targets = states[1:]
    
    # Generate random reservoir weights
    reservoir_size = 1000
    connectivity = 0.1  # Sparsity of the reservoir connections
    reservoir_weights = sparse_random(reservoir_size, reservoir_size, density=connectivity).toarray()
    
    # Generate random input weights
    input_weights = np.random.rand(reservoir_size, inputs.shape[1])
    
    # Generate random bias weights
    bias_weights = np.random.rand(reservoir_size)
    
    # Initialize reservoir states
    reservoir_states = np.zeros((inputs.shape[0], reservoir_size))
    
    # Compute reservoir states
    for i in range(1, inputs.shape[0]):
        reservoir_states[i] = np.tanh(np.dot(inputs[i], input_weights.T) +
                                      np.dot(reservoir_states[i-1], reservoir_weights) +
                                      bias_weights)

    
    # Train the readout layer using Ridge regression
    readout_model = Ridge(alpha=0.01)
    readout_model.fit(reservoir_states, targets)
    
    # Generate predictions using the ESN
    predicted_states = readout_model.predict(reservoir_states)
    
    # Compute mean square error (MSE)
    mse = mean_squared_error(targets, predicted_states)
    print(f"MSE (ρ = {rho}): {mse:.6f}")
    
    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], label='Actual')
    ax.plot(predicted_states[:, 0], predicted_states[:, 1], predicted_states[:, 2], label='Predicted')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Lorenz Equations (ρ = {rho})')
    ax.legend()
    plt.show()


# In[ ]:




