# EE-399: Introduction to Machine Learning
#### Applications of Neural Networks in Prediction
#### Sabrina Hwang

## Abstract:
This assignment explores the application of neural networks (NN) in predicting the future states of the Lorenz equations. The Lorenz equations are a set of nonlinear differential equations that describe the behavior of a simplified atmospheric convection model. The objective of this research is to train a NN to advance the solution from time $t$ to $t + ∆t$ for different values of the control parameter $ρ$ ($ρ = 10$, $28$, and $40$) and assess its performance in predicting future states for previously unexplored values of $ρ$ ($ρ = 17$ and $ρ = 35$).

To achieve this, various types of NN architectures are compared, namely feed-forward, Long Short-Term Memory (LSTM), Recurrent Neural Network (RNN), and Echo State Networks (ESN). Each architecture is trained using the given code provided in class emails, which simulates the Lorenz equations. The training process involves optimizing the NN parameters to minimize the prediction error and maximize the accuracy of future state forecasts.

Overall, the assignment aims to teach students how to apply NNs to advance solutions and predict future states in dynamic systems. It also provides an opportunity to assess and compare the performance of different NN architectures for forecasting the dynamics of complex systems, fostering an understanding of their respective strengths and weaknesses. This exercise enhances students' knowledge of neural network applications and equips them with valuable skills in modeling and predicting nonlinear and chaotic systems.

## Table of Contents:
* [Abstract](https://github.com/hwangsab/EE-399-HW5/blob/main/README.md#abstract)
* [Introduction and Overview](https://github.com/hwangsab/EE-399-HW5/blob/main/README.md#introduction-and-overview)
* [Theoretical Background](https://github.com/hwangsab/EE-399-HW5/blob/main/README.md#theoretical-background)
* [Algorithm Implementation and Development](https://github.com/hwangsab/EE-399-HW5/blob/main/README.md#algorithm-implementation-and-development)
  * [Code Description](https://github.com/hwangsab/EE-399-HW5/blob/main/README.md#code-description)
    * [Problem 1.1: Training a NN on Lorenz equations](https://github.com/hwangsab/EE-399-HW5/tree/main#problem-11-training-a-nn-on-lorenz-equations)
    * [Problem 1.2: Validating the NN on Lorenz equations model on test data](https://github.com/hwangsab/EE-399-HW5/tree/main#problem-12-validating-the-nn-on-lorenz-equations-model-on-test-data)
    * [Problem 2.1: Training a feed-forward network for forecasting](https://github.com/hwangsab/EE-399-HW5/tree/main#problem-21-training-a-feed-forward-network-for-forecasting)
    * [Problem 2.2: Training a LSTM network for forecasting](https://github.com/hwangsab/EE-399-HW5/tree/main#problem-22-training-a-lstm-network-for-forecasting)
    * [Problem 2.3: Training a RNN network for forecasting](https://github.com/hwangsab/EE-399-HW5/tree/main#problem-23-training-a-rnn-network-for-forecasting)
    * [Problem 2.4: Training an Echo State Network for forecasting](https://github.com/hwangsab/EE-399-HW5/tree/main#problem-24-training-an-echo-state-network-for-forecasting)
* [Computational Results](https://github.com/hwangsab/EE-399-HW5/blob/main/README.md#computational-results)
  * [Usage](https://github.com/hwangsab/EE-399-HW5/blob/main/README.md#usage)
  * [Problem 1.1: Training a NN on Lorenz equations](https://github.com/hwangsab/EE-399-HW5/tree/main#problem-11-training-a-nn-on-lorenz-equations-1)
  * [Problem 1.2: Validating the NN on Lorenz equations model on test data](https://github.com/hwangsab/EE-399-HW5/tree/main#problem-12-validating-the-nn-on-lorenz-equations-model-on-test-data-1)
  * [Problem 2.1: Training a feed-forward network for forecasting](https://github.com/hwangsab/EE-399-HW5/tree/main#problem-21-training-a-feed-forward-network-for-forecasting-1)
  * [Problem 2.2: Training a LSTM network for forecasting](https://github.com/hwangsab/EE-399-HW5/tree/main#problem-22-training-a-lstm-network-for-forecasting-1)
  * [Problem 2.3: Training a RNN network for forecasting](https://github.com/hwangsab/EE-399-HW5/tree/main#problem-23-training-a-rnn-network-for-forecasting-1)
  * [Problem 2.4: Training an Echo State Network for forecasting](https://github.com/hwangsab/EE-399-HW5/tree/main#problem-24-training-an-echo-state-network-for-forecasting-1)
* [Summary and Conclusion](https://github.com/hwangsab/EE-399-HW5/blob/main/README.md#summary-and-conclusions)

## Introduction and Overview:
The provided assignment prompt focuses on two core topics: training neural networks (NNs) to advance solutions and predict future states of the Lorenz equations, and comparing different NN architectures for forecasting dynamics.

The first objective of the assignment is to train a NN using the given code to advance the solution from time $t to t + ∆t$ for three different values of the control parameter $ρ: 10, 28, and 40$. This involves optimizing the NN parameters to minimize prediction errors and maximize accuracy. Subsequently, the trained NN is evaluated to assess its performance in predicting future states for previously unexplored ρ values: 17 and 35. This task tests the generalization capability of the NN beyond the specific training scenarios.

The second part of the assignment focuses on comparing four different NN architectures: feed-forward networks, Long Short-Term Memory (LSTM) networks, Recurrent Neural Networks (RNNs), and Echo State Networks (ESNs). The comparison aims to evaluate their effectiveness in forecasting the dynamics of the Lorenz equations. Each architecture has distinct characteristics: feed-forward networks excel in capturing static mappings, LSTM and RNN architectures are designed for modeling temporal dependencies, and ESNs leverage reservoir computing techniques.

## Theoretical Background:

The Lorenz equations are defined as a system of three coupled nonlinear differential equations, representing the evolution of variables $x$, $y$, and $z$ over time. The specific form of the Lorenz equations is implemented in the code as the `lorenz` function. To solve the Lorenz equations for different values of the control parameter ρ, the code utilizes the odeint function from the scipy.integrate module. This function numerically integrates the differential equations over a specified time interval (`t_start` to `t_end`) with a given time step (`delta_t`).

For each value of $ρ$ specified in the `rho_values` list, the code generates random initial conditions and solves the Lorenz equations using `odeint`. The resulting states are then used to train a NN to predict the next state given the current state. The NN architecture is defined using the Sequential model from the `tensorflow.keras.models` module, and it consists of three fully connected layers (`Dense`) with `ReLU` activation. The model is compiled with the mean squared error (MSE) loss function and trained using the fit method.

After training, the NN is used to generate predictions for the states of the Lorenz equations. The predicted states are compared to the actual states, and the MSE is computed using the `mean_squared_error` function from the `sklearn.metrics` module. The MSE serves as a measure of the prediction accuracy. Finally, the results are visualized using the `matplotlib.pyplot` module. The code plots the actual states of the Lorenz equations and the predicted states for each value of $ρ$ in a 3D plot.

For the second half of the assignment, implementation involves generating training data, defining multiple models, training them, and testing their predictions. The models used are feed-forward, Long Short-Term Memory (LSTM), Recurrent Neural Network (RNN), and Echo State Networks (ESN). The models could be further described as follows:
* Feed-forward Neural Network (FNN): A feed-forward neural network processes information in one direction, from input to output, without feedback connections. It's commonly used for tasks like classification and regression, but may struggle with sequential data.
* Long Short-Term Memory (LSTM): LSTM is a recurrent neural network (RNN) architecture designed to capture long-term dependencies in sequential data. It uses memory cells and gating mechanisms to retain and selectively update information over time.
* Recurrent Neural Network (RNN): RNNs are neural networks that process sequential data by maintaining hidden states, enabling them to capture temporal dependencies. They are widely used for tasks like language modeling and speech recognition.
* Echo State Networks (ESN): ESNs are a type of RNN with a fixed, randomly initialized recurrent layer called the "reservoir." The reservoir transforms input data into a high-dimensional representation, which is then processed by a readout layer. ESNs offer a balance between computational efficiency and modeling power, suitable for tasks like time series prediction and chaotic systems modeling.

## Algorithm Implementation and Development:
Completion of this project and subsequent development and implementation of the algorithm was 
accomplished through Python as our primary programming language. 

### Code Description
The code uses the following libraries for Problem 1:
* `NumPy (np)`: Fundamental library for numerical computing in Python.
* `SciPy (odeint)`: Library for solving ordinary differential equations numerically.
* `Matplotlib (plt)`: Plotting library for creating visualizations in Python.
* `scikit-learn (sklearn)`: Machine learning library for various tasks and metrics.
* `TensorFlow (tf) with Keras`: Deep learning library with a high-level API for neural networks.
  
And the following additional libraries for Problem 2:
* `mean_squared_error (sklearn.metrics)`: Function for calculating the mean squared error metric.
* `LSTM (tensorflow.keras.layers)`: Layer type for Long Short-Term Memory units in neural networks.
* `SimpleRNN (tensorflow.keras.layers)`: Layer type for Simple Recurrent Neural Networks in neural networks.
* `Ridge (sklearn.linear_model)`: Linear regression model using Ridge regularization.
* `sparse_random (scipy.sparse)`: Function for generating a random sparse matrix.

#### Problem 1.1: Training a NN on Lorenz equations
First, we set up the time steps and parameters for the simulation. It defines the start and end time (`t_start` and `t_end`), the time step size (`delta_t`), and the number of steps (`num_steps`) based on the given values. We then have the code iterate over each ρ value in the rho_values list. For each $ρ$, it generates initial conditions randomly using `np.random.randn(3)`

The Lorenz equations are solved numerically using the `odeint` function from the SciPy library. The resulting states are stored in the states variable. A feed-forward neural network model is then created using the `Sequential` class from the TensorFlow Keras library. The model consists of two hidden layers with 64 units each, using the ReLU activation function, and an output layer with three units 

#### Problem 1.2: Validating the NN on Lorenz equations model on test data
The second half of the first problem builds upon the first half, and tests the neural network model generated on test data with different $ρ$ values. 
```
rho_values = [17, 35]
sigma = 10.0
beta = 8.0 / 3.0
```

#### Problem 2.1: Training a feed-forward network for forecasting
The second problem in the assignment assigns the comparison for different kinds of neural networks, beginning with feed-forward networks. In this code, a feed-forward neural network model is generated and trained for each value of $ρ$ in the given list. 

For each $ρ$ value, the code generates training data using the `generate_data` function with the specified parameters:
```
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
```
The input and output sequences are created by splitting the training data accordingly.

Then, a new model is created using the "create_model" function:
```
def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3))
    return model
```
The model is trained using the training data, with a specified number of epochs and batch size. After training, the model is tested by predicting the states at future time steps based on the last state in the training data. The predicted states are stored in the "predicted_data" array using a loop.

#### Problem 2.2: Training a LSTM network for forecasting
This section of the code is almost identical to that seen in 2.1, with the exception of a different network being used:
```
model = Sequential()
model.add(LSTM(64, input_shape=(1, 3), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')
model.fit(inputs, targets, epochs=10, batch_size=32)
```

#### Problem 2.3: Training a RNN network for forecasting
This section of the code is almost identical to that seen in 2.1, with the exception of a different network being used:
```
model = Sequential()
model.add(SimpleRNN(64, input_shape=(1, 3), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')
model.fit(inputs, targets, epochs=10, batch_size=32)
```

#### Problem 2.4: Training an Echo State Network for forecasting
This section of the code is a bit different from the rest, specifically that the setup of the Echo State Network is written differently:
```
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
```

## Computational Results:

### Usage
To run the code, simply run the Python file `EE399 HW5.py` in any Python environment. The output will be 
printed to the console and displayed in a pop-up window.

#### Problem 1.1: Training a NN on Lorenz equations
The data that was predicted from the Lorenz Equations were plotted as follows:     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/a2640e9a-cabb-40ce-95a1-3a9a0f29cd9a)     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/7be2f25a-cc61-4d32-8921-4cb8abc3262a)     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/2e0c2b5c-1704-422a-bcbb-af353ada2f30)     

In addition, the following MSEs were computed:
```
MSE (ρ = 10): 0.009927
MSE (ρ = 28): 0.018780
MSE (ρ = 40): 0.310694
```

#### Problem 1.2: Validating the NN on Lorenz equations model on test data
The data that was predicted from the Lorenz Equations were plotted as follows:     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/2ffdfdeb-5f69-4032-9c27-c33578cdd9bf)     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/c68e7f9a-c42a-475f-9395-21fd621df918)     

#### Problem 2.1: Training a feed-forward network for forecasting
The data that was predicted from the Lorenz Equations were plotted as follows:     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/83837d90-5075-415e-909e-f87c2f51d64c)     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/e69df91d-d668-44f7-93ed-5761be97ec67)     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/c7639dec-24c8-4b58-b33e-232639007480)     

In addition, the following MSEs were computed:
```
MSE (ρ = 10): 0.009111
MSE (ρ = 28): 0.159188
MSE (ρ = 40): 0.395996
```
#### Problem 2.2: Training a LSTM network for forecasting
The data that was predicted from the Lorenz Equations were plotted as follows:     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/7f7fb700-bffa-4a71-aaec-e6bd5471939f)     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/2664a75e-5378-407b-b5f6-5db51f67742e)     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/cd078442-0354-4214-876d-f99f81a958aa)     

In addition, the following MSEs were computed:
```
MSE (ρ = 10): 0.039650
MSE (ρ = 28): 0.134333
MSE (ρ = 40): 0.155343
```
#### Problem 2.3: Training a RNN network for forecasting
The data that was predicted from the Lorenz Equations were plotted as follows:     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/d2f7df06-1e19-491c-bf4e-56687603edfd)     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/c979d24d-d859-44fb-83a5-e4c88fed2965)     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/a6b9e8f4-6b45-4252-bd75-71e6feaef958)     

In addition, the following MSEs were computed:
```
MSE (ρ = 10): 0.009638
MSE (ρ = 28): 0.023434
MSE (ρ = 40): 0.299039
```
#### Problem 2.4: Training an Echo State Network for forecasting
The data that was predicted from the Lorenz Equations were plotted as follows:     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/ecdc3f38-c0cf-4170-9faf-93cfb6b9cd10)     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/55623eab-6e96-4582-9b5e-13a1142ad965)     
![download](https://github.com/hwangsab/EE-399-HW5/assets/125385468/a1152556-bfa2-4b37-99ae-954261d144e3)     

In addition, the following MSEs were computed:
```
MSE (ρ = 10): 2.152644
MSE (ρ = 28): 21.434330
MSE (ρ = 40): 115.116369
```

Overall, among the neural network architectures considered, the feed-forward network achieves the lowest MSE values for all values of $ρ$. This indicates that the feed-forward network generally performs better in capturing the underlying dynamics of the system and predicting future states. Similarly, the LSTM network also performs well, exhibiting relatively low MSE values across different values of $ρ$. The RNN network follows shortly after in terms of performance. 

## Summary and Conclusions:
The assignment covered a range of topics related to neural networks and their applications. In order to cover the different network architectures, such as feed-forward, LSTM, RNN, and Echo State, we generated and fit these networks to the same sample data to compute their mean square errors, and to understand the strengths and weaknesses of each one. 

Through this assignment, we gained valuable insights into the strengths and limitations of various network architectures. Overall, among the neural network architectures considered, the feed-forward network achieves the lowest MSE values for all values of $ρ$. In conclusion, this assignment showcased the strengths and variabilities that different kinds of neural networks offer when predicting values, within the context of Lorenz equations. It highlighted the importance of selecting appropriate models for specific tasks and evaluating their performance using suitable metrics like MSE. 
