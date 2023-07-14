# EE-399: Introduction to Machine Learning
#### Application of LSTM decoder models
#### Sabrina Hwang

## Abstract:
This assignment explores the use of an existing LSTM/decoder model for analyzing sea-surface temperature data. It calls upon the existing example code and data from a GitHub repository (https://github.com/Jan-Williams/pyshred). After training and plotting the results from the existent model, we're tasked to analyze the performance of the model with different variables that may affect the overall abilities of the model, such as time lag, Gaussian noise, and sensor quanity. 

## Table of Contents:
* [Abstract](https://github.com/hwangsab/EE-399-HW6/blob/main/README.md#abstract)
* [Introduction and Overview](https://github.com/hwangsab/EE-399-HW6/blob/main/README.md#introduction-and-overview)
* [Theoretical Background](https://github.com/hwangsab/EE-399-HW6/blob/main/README.md#theoretical-background)
* [Algorithm Implementation and Development](https://github.com/hwangsab/EE-399-HW6/blob/main/README.md#algorithm-implementation-and-development)
  * [Code Description](https://github.com/hwangsab/EE-399-HW6/blob/main/README.md#code-description)
    * [Problem 1.1 & 1.2: Training an existing LSTM/decoder model and plotting results]()
    * [Problem 1.3: Analyzing performance as a function of time lag]()
    * [Problem 1.4: Analyzing performance as a function of noise]()
    * [Problem 1.5: Analyzing performance as a function of sensor quanity]()
* [Computational Results](https://github.com/hwangsab/EE-399-HW6/blob/main/README.md#computational-results)
  * [Usage](https://github.com/hwangsab/EE-399-HW6/blob/main/README.md#usage)
  * [Problem 1.1 & 1.2: Training an existing LSTM/decoder model and plotting results]()
  * [Problem 1.3: Analyzing performance as a function of time lag]()
  * [Problem 1.4: Analyzing performance as a function of noise]()
  * [Problem 1.5: Analyzing performance as a function of sensor quanity]()
* [Summary and Conclusion](https://github.com/hwangsab/EE-399-HW6/blob/main/README.md#summary-and-conclusions)

## Introduction and Overview:
The assignment begins with asking for us to train the LSTM/decoder model using the provided example code and dataset. The model is designed to capture temporal dependencies and generate accurate predictions based on sea-surface temperature data. After training the model, the results are plotted to visualize the predicted temperature values and compare them with the actual values.

Next, the assignment focuses on analyzing the performance of the model with respect to the time lag variable. By systematically varying the time lag between input and output data, the impact on the model's predictive ability is examined. This analysis provides insights into the optimal time lag for accurate predictions and helps understand the temporal dynamics of sea-surface temperature fluctuations.

We then explore the effect of noise on the model's performance. Gaussian noise is added to the sea-surface temperature data, simulating real-world scenarios where measurements may be subject to various sources of noise. By analyzing the model's ability to handle noisy data, the assignment assesses its robustness and generalizability in the presence of noise.

Lastly, we investigate the performance of the model in relation to the number of sensors. Different configurations with varying numbers of sensors are tested to evaluate how the model adapts and performs under different levels of sensor availability. This analysis provides insights into the scalability and reliability of the model when the number of sensors is altered.

## Theoretical Background:
This theoretical background discusses the concepts of LSTM/decoder models, Gaussian noise, time lag, and sensor number, and their potential impact on the performance of the LSTM/decoder model in the context of sea-surface temperature analysis.

LSTM/decoder models are a type of recurrent neural network (RNN) architecture that excels in modeling sequential data. The decoder component in the LSTM/decoder model allows the model to generate predictions based on the learned temporal patterns. By training the LSTM/decoder model on historical sea-surface temperature data, it can learn to capture complex patterns and make accurate predictions of future temperature values.

Gaussian noise is a type of random noise that follows a Gaussian distribution. In the context of sea-surface temperature analysis, Gaussian noise can represent various sources of measurement errors or environmental disturbances. By adding Gaussian noise to the input data, the model's robustness and ability to handle noisy inputs can be evaluated. This analysis provides insights into the model's generalizability in real-world scenarios where data may be subject to uncertainties.

Time lag refers to the temporal gap or delay between input and output data points. In sea-surface temperature analysis, the time lag variable represents the time interval between past temperature observations and the predicted future temperature value. The choice of an appropriate time lag is crucial as it affects the model's ability to capture short-term or long-term temporal dependencies. By systematically varying the time lag, the model's performance can be evaluated to determine the optimal time lag for accurate predictions. Understanding the impact of time lag helps in understanding the dynamics of sea-surface temperature fluctuations and selecting an appropriate temporal context for prediction.

The number of sensors used in sea-surface temperature analysis can significantly impact the performance of the LSTM/decoder model. Different sensor configurations, representing varying spatial coverage and density, can affect the model's ability to capture spatial patterns and generalize across different regions. By varying the number of sensors, the model's scalability and reliability can be assessed. This analysis provides insights into the trade-off between sensor availability and the model's predictive performance, facilitating the design of efficient monitoring systems and data assimilation strategies.

## Algorithm Implementation and Development:
Completion of this project and subsequent development and implementation of the algorithm was accomplished through Python as our primary programming language. 

### Code Description
The code uses the following libraries for Problem 1:
* `import numpy as np`: Imports the NumPy library and assigns it the alias "np" for ease of use. 
* `from processdata import load_data`: Imports the function "load_data" from the "processdata" module. 
* `from processdata import TimeSeriesDataset`: Imports the class "TimeSeriesDataset" from the "processdata" module. 
* `import models`: Imports the "models" module. 
* `import torch`: Imports the PyTorch library. 
* `import matplotlib.pyplot as plt`: Imports the Pyplot module from the Matplotlib library and assigns it the alias "plt" for ease of use. 
* `from sklearn.preprocessing import MinMaxScaler`: Imports the "MinMaxScaler" class from the "preprocessing" module within the "sklearn" library. 

#### 1.1 & 1.2: Training an existing LSTM/decoder model and plotting results
For downloading the example code that was provided for this assignment, we used the git clone command available on python as seen, and then changed the working directory. 
```
!git clone https://github.com/Jan-Williams/pyshred
%cd pyshred
```

From that point, training the SHRED model was computed by calling the imported models and fitting them with the appropriate parameters:
```
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3,
                               verbose=False, patience=5)
```

And then the model performance was evaluated by calculating the error between the original and the reconstructed data. 
```
test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
error = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
```

#### Problem 1.3: Analyzing performance as a function of time lag
This problem asked for the analysis of the performance as a function of the time lag variable. The code in the earlier cells actually address Problem 1.3, but pause midway to graph the original and reconstructed data. 

After designating different time lag values:
```
lag_values = [10, 20, 30, 40, 50, 60]
```

We call on the `train_model_with_lags` function and calculate the error in the same way as determined above. 
```
for lag in lag_values:
    error = train_model_with_lags(lag)
    errors.append(error)
    print(f"Lag: {lag}, Error: {error}")
```

#### Problem 1.4: Analyzing performance as a function of noise
For Problem 1.4, we write a new function that creates and adds Gaussian noise to the existing data as seen below: 
```
def add_gaussian_noise(data, mean=0, std=0.1):
    noise = np.random.normal(mean, std, size=data.shape)
    noisy_data = data + noise
    return noisy_data
```

We then added the generated Gaussian noise to the data by calling upon the function in a later line,
```
noisy_X = add_gaussian_noise(transformed_X, std=noise_std)
```
And then trained the SHRED model and evaluated its performance the same way we had implemented for the prior problems. 

#### Problem 1.5: Analyzing performance as a function of sensor quanity
Problem 1.5 bears a lot of simiarity to Problem 1.3, in the sense that a variable `sensor_values` stores a test number of sensors that we would like to compute: 
```
sensor_values = [2, 3, 4, 5, 6]
```

And then we call a slightly modified version of `train_model_with_lags` function called `train_model_with_num_sensors` and compute the error in the same way. 
```
for num_sensors in sensor_values:
    error = train_model_with_num_sensors(num_sensors)
    errors.append(error)
    print(f"Number of Sensors: {num_sensors}, Error: {error}")
```

## Computational Results:

### Usage
To run the code, simply run the Python file `EE399 HW6.py` in any Python environment. The output will be 
printed to the console and displayed in a pop-up window.

#### Problem 1.1 & 1.2: Training an existing LSTM/decoder model and plotting results: 
The plotted results can be viewed as followed:   
![download](https://github.com/hwangsab/EE-399-HW6/assets/125385468/ad2f2b7c-618e-48d8-991c-5f01bf58e88f)

#### Problem 1.3: Analyzing performance as a function of time lag
The performance as a function of time lag is computed into the plot as seen:   
![download](https://github.com/hwangsab/EE-399-HW6/assets/125385468/274c89ef-ce48-42c9-a21e-9d93c4821394)

And the individual values were computed to be:   
```
Lag: 10, Error: 0.02353539690375328
Lag: 20, Error: 0.02207941934466362
Lag: 30, Error: 0.01973891258239746
Lag: 40, Error: 0.01969754882156849
Lag: 50, Error: 0.020615896210074425
Lag: 60, Error: 0.019928231835365295
```

The results indicate that as the time lag increases, the model's prediction error tends to decrease. This suggests that the model benefits from a longer temporal context when capturing the underlying dynamics of sea-surface temperature.

#### Problem 1.4: Analyzing performance as a function of noise
The performance as a function of noise is computed into the plot as seen:   
![download](https://github.com/hwangsab/EE-399-HW6/assets/125385468/72d2c321-e1ee-452c-9881-5b4e218a6218)

And the individual values were computed to be:   
```
Noise Std: 0.01, Error: 0.05266956239938736
Noise Std: 0.05, Error: 0.049338869750499725
Noise Std: 0.1, Error: 0.048108555376529694
Noise Std: 0.2, Error: 0.07029640674591064
Noise Std: 0.5, Error: 0.09328985959291458
```

The results indicate that as the standard deviation of the noise increases, the model's performance degrades. Higher levels of noise introduce more uncertainty and make it challenging for the model to accurately predict sea-surface temperature values. Therefore, minimizing noise sources and employing noise reduction techniques are crucial for improving the model's performance.

#### Problem 1.5: Analyzing performance as a function of sensor quantity
The performance as a function of sensor quantity is computed into the plot as seen:   
![download](https://github.com/hwangsab/EE-399-HW6/assets/125385468/07858091-45b8-41e7-a777-c7dc9820fb00)

And the individual values were computed to be:   
```
Number of Sensors: 2, Error: 0.02079629711806774
Number of Sensors: 3, Error: 0.01986599899828434
Number of Sensors: 4, Error: 0.02045321650803089
Number of Sensors: 5, Error: 0.02035946026444435
Number of Sensors: 6, Error: 0.019188588485121727
```

The results suggest that increasing the number of sensors has a marginal effect on the model's performance. The model exhibits consistent accuracy across different sensor configurations, indicating its ability to capture spatial patterns effectively. However, further analysis may be required to investigate the optimal sensor density and spatial coverage for specific applications.

## Summary and Conclusions:
Overall, this homework assignment offers a comprehensive exploration of an LSTM/decoder model applied to sea-surface temperature analysis. Through training, result plotting, and various analyses, the assignment investigates the model's performance with respect to time lag, noise, and the number of sensors. The findings provide valuable insights into the model's capabilities and limitations, contributing to the understanding and application of LSTM-based approaches in analyzing environmental data.

In conclusion, the analysis of the LSTM/decoder model's performance provides valuable insights into its behavior in sea-surface temperature analysis. The findings suggest that increasing the time lag improves prediction accuracy, while higher levels of noise degrade performance. The model demonstrates robustness to different sensor configurations, with consistent accuracy observed across varying numbers of sensors. These results contribute to the understanding and optimization of LSTM/decoder models in environmental data analysis and inform the design of monitoring systems for sea-surface temperature monitoring.
