# -*- coding: utf-8 -*-
"""EE399 HW6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1A5jOEYe7tpfgcYB8XFgIKhCmU8Jl5mrY

# EE 399 Introduction to Machine Learning: HW 6 Submission
### Sabrina Hwang
"""

# GitHub HW6: https://github.com/hwangsab/EE-399-HW6

!git clone https://github.com/Jan-Williams/pyshred

# Commented out IPython magic to ensure Python compatibility.
# %cd pyshred

import numpy as np
from processdata import load_data
from processdata import TimeSeriesDataset
import models
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""Problem 1.1 & 1.2: Download the example code (and data) for sea-surface temperature which uses an LSTM/decoder, and train the model and plot the results"""

def train_model_with_lags(lags):
    # Load and preprocess data
    num_sensors = 3
    load_X = load_data('SST')
    n = load_X.shape[0]
    m = load_X.shape[1]
    sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

    train_indices = np.random.choice(n - lags, size=1000, replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    valid_test_indices = np.arange(0, n - lags)[np.where(mask != 0)[0]]
    valid_indices = valid_test_indices[::2]
    test_indices = valid_test_indices[1::2]

    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)

    # Generate input sequences to a SHRED model
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i + lags, sensor_locations]

    # Generate training, validation, and test datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    # Train SHRED model
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3,
                                   verbose=False, patience=5)

    # Evaluate model performance on the test dataset
    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    error = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)

    # Plot original vs. reconstructed data
    plotdata = [test_ground_truth, test_recons]
    labels = ['Original Data', 'Reconstructed Data']
    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 4))

    for axis, p, label in zip(ax, plotdata, labels):
        im = axis.imshow(np.flipud(p[0]), origin='lower')  # Use np.flipud() to flip the plot vertically
        axis.set_aspect('equal')
        axis.set_xticks([])
        axis.set_yticks([])

        # Set custom tick labels and spacing for the y-axis
        axis.set_ylim(0, p[0].shape[0])
        axis.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        axis.tick_params(axis='y', pad=8, labelsize=8)

        # Set title for each plot
        axis.set_title(label, fontsize=12)

    # Add x-label, y-label, and plot title
    fig.text(0.5, 0.02, 'Time', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Value', va='center', rotation='vertical', fontsize=12)

    # Add colorbar for the plot
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.5])  # Adjust height of colorbar
    fig.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.show()

    return error

"""Problem 1.3: Do an analysis of the performance as a function of the time lag variable"""

# Perform analysis for different time lag values
lag_values = [10, 20, 30, 40, 50, 60]
errors = []

for lag in lag_values:
    error = train_model_with_lags(lag)
    errors.append(error)
    print(f"Lag: {lag}, Error: {error}")
    
# Plot the performance as a function of the time lag variable
plt.plot(lag_values, errors, marker='o')
plt.xlabel('Time Lag')
plt.ylabel('Performance Error')
plt.title('Performance vs. Time Lag')
plt.show()

"""Problem 4: Do an analysis of the performance as a function of noise (add Gaussian noise to data)"""

def add_gaussian_noise(data, mean=0, std=0.1):
    noise = np.random.normal(mean, std, size=data.shape)
    noisy_data = data + noise
    return noisy_data

def train_model_with_noise(noise_std):
    # Load and preprocess data
    num_sensors = 3
    load_X = load_data('SST')
    n = load_X.shape[0]
    m = load_X.shape[1]
    sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

    train_indices = np.random.choice(n, size=1000, replace=False)
    mask = np.ones(n)
    mask[train_indices] = 0
    valid_test_indices = np.arange(n)[np.where(mask != 0)[0]]
    valid_indices = valid_test_indices[::2]
    test_indices = valid_test_indices[1::2]

    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)

    # Add Gaussian noise to the data
    noisy_X = add_gaussian_noise(transformed_X, std=noise_std)

    # Generate input sequences to a SHRED model
    all_data_in = np.zeros((n, 1, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = noisy_X[i, sensor_locations].reshape(1, -1)

    # Generate training, validation, and test datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    train_data_out = torch.tensor(transformed_X[train_indices], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    # Train SHRED model
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3,
                                   verbose=False, patience=5)

    # Evaluate model performance on the test dataset
    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    error = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)

    return error

# Perform analysis for different noise levels
noise_std_values = [0.01, 0.05, 0.1, 0.2, 0.5]
errors = []

for std in noise_std_values:
    error = train_model_with_noise(std)
    errors.append(error)
    print(f"Noise Std: {std}, Error: {error}")

# Plot the performance as a function of noise
plt.plot(noise_std_values, errors, marker='o')
plt.xlabel('Noise Standard Deviation')
plt.ylabel('Performance Error')
plt.title('Performance vs. Noise')
plt.show()

"""Problem 1.5: Do an analysis of the performance as a function of the number of sensors

"""

def train_model_with_num_sensors(num_sensors):
    # Load and preprocess data
    lags = 52
    load_X = load_data('SST')
    n = load_X.shape[0]
    m = load_X.shape[1]
    sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

    train_indices = np.random.choice(n - lags, size=1000, replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    valid_test_indices = np.arange(0, n - lags)[np.where(mask != 0)[0]]
    valid_indices = valid_test_indices[::2]
    test_indices = valid_test_indices[1::2]

    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)

    # Generate input sequences to a SHRED model
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i + lags, sensor_locations]

    # Generate training, validation, and test datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    # Train SHRED model
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3,
                                   verbose=False, patience=5)

    # Evaluate model performance on the test dataset
    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    error = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)

    return error

# Perform analysis for different number of sensors
sensor_values = [2, 3, 4, 5, 6]
errors = []

for num_sensors in sensor_values:
    error = train_model_with_num_sensors(num_sensors)
    errors.append(error)
    print(f"Number of Sensors: {num_sensors}, Error: {error}")

# Plot the performance as a function of the number of sensors
plt.plot(sensor_values, errors, marker='o')
plt.xlabel('Number of Sensors')
plt.ylabel('Performance Error')
plt.title('Performance vs. Number of Sensors')
plt.show()