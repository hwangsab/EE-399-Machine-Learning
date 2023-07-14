# EE-399: Introduction to Machine Learning
#### Application of Machine Learning Models for Classification
#### Sabrina Hwang

## Abstract:
The assignment focuses on the implementation of various machine learning models for digit classification using the MNIST dataset. It consists of two parts: Part I and Part II.

In Part I, the code focuses on a specific dataset consisting of numerical features and corresponding labels. The dataset is preprocessed by normalizing the input features. The first model is a regression model that predicts a numerical target variable. The dataset is split into training and test sets, and a neural network with multiple layers is constructed. The model is trained on the training set and evaluated on the test set using the mean squared error metric. Next, the dataset is split into training and test sets in a different manner. A regression model is trained on the training set and evaluated on both the training and test sets using the mean squared error metric. Lastly, the dataset is split into training and test sets with a different configuration. A regression model is trained on the training set and evaluated on both the training and test sets using the mean squared error metric.

In Part II, the code demonstrates the use of different models such as a feed-forward neural network, an LSTM classifier, an SVM classifier, and a decision tree classifier. The MNIST dataset is loaded, and the features and labels are extracted. PCA is applied to reduce the dimensionality of the data, and the first 20 PCA modes are computed along with their explained variance ratios. The feed-forward neural network is implemented with a sequential architecture and trained on the normalized features. The LSTM classifier is built to process the sequential nature of the input data. The SVM classifier is trained using the Support Vector Machine algorithm and tested on the test set to assess its accuracy. Lastly, the decision tree classifier is implemented, trained, and evaluated on the test set.

## Table of Contents:
* [Abstract](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#abstract)
* [Introduction and Overview](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#introduction-and-overview)
* [Theoretical Background](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#theoretical-background)
* [Algorithm Implementation and Development](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#algorithm-implementation-and-development)
  * [Code Description](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#code-description)
    * [Part I Problem (i): Fitting data to a three layer feed-forward neural network](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#part-i-problem-i-fitting-data-to-a-three-layer-feed-forward-neural-network)
    * [Part I Problem (ii): Training data to a neural network and computing LSE](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#part-i-problem-ii-training-data-to-a-neural-network-and-computing-lse)
    * [Part I Problem (iii): Training data to a neural network and computing LSE](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#part-i-problem-iii-training-data-to-a-neural-network-and-computing-lse)
    * [Part I Problem (iv): Comparison between linear regression vs. neural network models](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#part-i-problem-iv-comparison-between-linear-regression-vs-neural-network-models)
    * [Part II Problem (i): PCA modes on the MNIST dataset](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#part-ii-problem-i-pca-modes-on-the-mnist-dataset)
    * [Part II Problem (ii): Neural networks, LSTM, SVM, and decision tree classifiers](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#part-ii-problem-ii-neural-networks-lstm-svm-and-decision-tree-classifiers)
* [Computational Results](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#computational-results)
  * [Usage](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#usage)
  * [Part I Problem (i): Fitting data to a three layer feed-forward neural network](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#part-i-problem-i-fitting-data-to-a-three-layer-feed-forward-neural-network-1)
  * [Part I Problem (ii): Training data to a neural network and computing LSE](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#part-i-problem-ii-training-data-to-a-neural-network-and-computing-lse-1)
  * [Part I Problem (iii): Training data to a neural network and computing LSE](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#part-i-problem-iii-training-data-to-a-neural-network-and-computing-lse-1)
  * [Part I Problem (iv): Comparison between linear regression vs. neural network models](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#part-i-problem-iv-comparison-between-linear-regression-vs-neural-network-models-1)
  * [Part II Problem (i): PCA modes on the MNIST dataset](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#part-ii-problem-i-pca-modes-on-the-mnist-dataset-1)
  * [Part II Problem (ii): Neural networks, LSTM, SVM, and decision tree classifiers](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#part-ii-problem-ii-neural-networks-lstm-svm-and-decision-tree-classifiers-1)
* [Summary and Conclusion](https://github.com/hwangsab/EE-399-HW4/blob/main/README.md#summary-and-conclusions)

## Introduction and Overview:
The first portion of this assignment demonstrates the implementation of a simple regression model using TensorFlow to predict the values of a target variable based on a given input. The goal of this code is to showcase how to build, train, and evaluate a neural network model for regression tasks. 

The second portion of this assignment demonstrates the implementation of various machine learning models for digit classification using the MNIST dataset. The MNIST dataset is a widely used benchmark dataset in the field of machine learning and consists of handwritten digits.

## Theoretical Background:

The theoretical foundation for Part I of the code is based on the concept building and training neural network models for regression. The code begins by defining a dataset consisting of input values (`X`) and corresponding target values (`Y`). In this case, `X` represents a range of numbers from 0 to 30, while `Y` contains a set of corresponding values. Next, the input data is normalized by subtracting the mean and dividing by the standard deviation, creating X_normalized. This step ensures that the input features are scaled appropriately for training the model. 

The model architecture is then defined using the TensorFlow's Sequential API. The model aims to learn a mapping between the input values and the target variable. The model is trained using the `fit` method, where the input (`X_normalized`) and target (`Y`) data are provided along with the desired number of epochs. 

After training, the model is used to make predictions on the same input range (`X_test_normalized`), and the results are printed alongside the actual target values. The model is redefined, compiled, and trained on the training data. The predictions are then obtained for both the training and test data. Finally, the least square error is computed for both the training and test data by comparing the predictions with the corresponding true values. The resulting errors are printed to evaluate the model's performance.

The code for Part II starts by loading the MNIST dataset using the `fetch_openml` function from `scikit-learn`. The dataset is then split into features (`X`) and labels (`y`), where `X` represents the pixel values of the digit images, and `y` represents the corresponding digit labels. Next, the features (`X`) are normalized by dividing them by 255.0 to scale the pixel values between 0 and 1.0. This normalization step ensures that the input data is in a suitable range for training the models. The code then proceeds to perform Principal Component Analysis (PCA) on the normalized features. 

PCA is applied to reduce the dimensionality of the data while retaining most of the information. In this case, PCA is performed with 20 components. The resulting transformed features (`X_pca`) are printed, and the first 20 PCA modes and their explained variance ratios are displayed. 

The first model implemented is a feed-forward neural network. The features and labels are split into training and test sets using the `train_test_split` function. The neural network architecture consists of three dense layers with varying activation functions. The model's performance is evaluated on the test set, and the test loss and accuracy are printed. 

The next model implemented is an LSTM (Long Short-Term Memory) classifier. The features and labels are again split into training and test sets, and the input data is reshaped to fit the LSTM input shape. The LSTM classifier architecture is defined with an LSTM layer and two dense layers. The model is compiled, trained, and evaluated similarly to the feed-forward neural network. The code then proceeds to build an SVM (Support Vector Machine) classifier. The features and labels are split into training and test sets. The SVM classifier is initialized, trained on the training data, and evaluated on the test set. Finally, a decision tree classifier is implemented. The labels are converted to one-hot encoded vectors, and the features and labels are split into training and test sets. The decision tree classifier is built, trained, and evaluated on the test set. This code provides an overview of different classifiers implemented on the MNIST dataset, allowing for a comparison of their performance in digit classification tasks.

## Algorithm Implementation and Development:
This homework assignment works around the MNIST dataset loaded through the following lines of code:
```
mnist = fetch_openml('mnist_784', parser='auto')
X = mnist.data / 255.0  # Scale the data to [0, 1]
y = mnist.target
```

Completion of this project and subsequent development and implementation of the algorithm was 
accomplished through Python as our primary programming language. 

### Code Description
The code uses the following libraries for Part I:
* `numpy`: Library for numerical computing with arrays and mathematical functions.
* `tensorflow`: Open-source machine learning framework for building neural networks.
* `sklearn.model_selection.train_test_split`: Function to split a dataset into training and testing subsets.
  
And the following additional libraries for Part II:
* `sklearn.datasets.fetch_openml`: Function to retrieve datasets from the OpenML platform.
* `tensorflow.keras`: High-level API for building and training neural networks with TensorFlow.
* `sklearn.decomposition.PCA`: Class for performing Principal Component Analysis, a dimensionality reduction technique.
* `sklearn.preprocessing.MinMaxScaler`: Class for scaling features to a specified range using min-max normalization.
* `sklearn.preprocessing.LabelBinarizer`: Class for converting categorical labels into binary vectors.
* `sklearn.model_selection.train_test_split`: Function to split a dataset into training and testing subsets.
* `sklearn.svm.SVC`: Class for implementing Support Vector Classification, a supervised learning algorithm for classification.
* `sklearn.tree.DecisionTreeClassifier`: Class for building decision tree-based classification models.
* `numpy (imported as np)`: Library for numerical computing with arrays and mathematical functions.
* `tensorflow (imported as tf)`: Open-source machine learning framework for building neural networks.

#### Part I Problem (i): Fitting data to a three layer feed-forward neural network
First, the input data is normalized to ensure consistent scaling across features. The model consists of two hidden layers, each with 64 units. The input shape is set to `(1,),` indicating that the model expects one-dimensional input. The final layer has a single unit. The model definition is as follows:
```
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```
The model is trained using the normalized input data `X_normalized` and the corresponding target values `Y`. The training is performed for 1000 epochs, and the verbose parameter is set to 0 to suppress the training progress updates. Once the model is trained, it is utilized to make predictions on a test range of data (`X_test`). The test data is normalized using the same normalization formula as before. The predictions are obtained using the predict method:
```
X_test = np.arange(0, 31)
X_test_normalized = (X_test - np.mean(X)) / np.std(X)
predictions = model.predict(X_test_normalized)
```

#### Part I Problem (ii): Training data to a neural network and computing LSE
First, the data is split into training and test sets. The final layer has a single unit:
```
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```
Next, the model is trained on the training data using the fit method. The training is performed for 1000 epochs, and the verbose parameter is set to 0 to suppress the training progress updates. After training, the trained model is used to make predictions on both the training and test data. 

The least square errors are then computed for both the training and test data by comparing the predictions with the corresponding target values:
```
lsq_error_train = np.mean((predictions_train.flatten() - Y_train) ** 2)
lsq_error_test = np.mean((predictions_test.flatten() - Y_test) ** 2)
```

#### Part I Problem (iii): Training data to a neural network and computing LSE
Problem (iii) asks for a similar process, only now the data split from training and test is different. The first 10 samples and the last 10 samples are concatenated to form the training data, while the samples from index 10 to 20 are used as the test data:
```
X_train = np.concatenate((X_normalized[:10], X_normalized[-10:]))
Y_train = np.concatenate((Y[:10], Y[-10:]))
X_test = X_normalized[10:20]
Y_test = Y[10:20]
```
The remaining components of the code are exactly the same as that seen in Problem (ii)

#### Part I Problem (iv): Comparison between linear regression vs. neural network models
The third problem asks to compare the models fit in homework one to the neural networks in (ii) and (iii). There is no explicit code to answer this problem.

#### Part II Problem (i): PCA modes on the MNIST dataset
The code performs Principal Component Analysis (PCA) on the input data `X` to reduce its dimensionality to 20 components. The algorithm, which has been implemented in past homework assignments can be observed as follows:
```
# Compute PCA with 20 components
n_components = 20
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
```

The PCA class from the sklearn.decomposition module is used to perform PCA. The number of components is set to 20, and the `fit_transform` method is called to compute the transformed data `X_pca`. This step reduces the dimensionality of the input data while retaining the most important information. 

The `explained_variance_ratio_` attribute of the pca object is used to access the variance ratio explained by each principal component. This ratio represents the proportion of the dataset's variance that is captured by each component. The `formatted_variance_ratio` list is created to store the formatted explained variance ratios. Each ratio is converted to a percentage format with two decimal places.

#### Part II Problem (ii): Neural networks, LSTM, SVM, and decision tree classifiers
The code implements multiple classification algorithms on the given dataset. It starts with a feed-forward neural network, followed by an LSTM classifier, an SVM classifier, and finally a decision tree classifier.

The input data `X` and corresponding labels `y` are split into training and test sets using the `train_test_split` function from the `sklearn.model_selection` module. The test set size is set to 20% of the data.
```
# Define the architecture of the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```
A feed-forward neural network model is defined using the Sequential class from `tf.keras.models`. The model has three dense layers with different activation functions. The input shape is set to 784, corresponding to the number of features in the dataset. The model is trained on the training data for 10 epochs with a batch size of 32. The validation data `(X_test, y_test)` is used to evaluate the model's performance during training.

The code continues with similar patterns for the LSTM classifier, SVM classifier, and decision tree classifier. It splits the data, defines the model architecture, compiles the model, trains the model, evaluates the model, and prints the test accuracy.

The LSTM classifier reshapes the input data for LSTM and uses LSTM and dense layers for classification. The SVM classifier utilizes the SVC class from sklearn.svm and directly trains and evaluates the classifier. The decision tree classifier employs the DecisionTreeClassifier class from sklearn.tree and trains and evaluates the classifier. This code showcases the implementation and evaluation of different classifiers, including a feed-forward neural network, LSTM classifier, SVM classifier, and decision tree classifier, on the given dataset.

## Computational Results:

### Usage
To run the code, simply run the Python file `EE399 HW4.py` in any Python environment. The output will be 
printed to the console and displayed in a pop-up window.

#### Part I Problem (i): Fitting data to a three layer feed-forward neural network
The data was fitted to a three layer feed forward neural network, where their predicted Y values were as projected:
```
X      Y      Predicted Y
--------------------
0      30     33.83460998535156
1      35     33.96746826171875
2      33     34.17804718017578
3      32     34.46626281738281
4      34     34.77323913574219
5      37     35.08021545410156
6      39     35.38719940185547
7      38     35.83856201171875
8      36     36.513084411621094
9      36     37.286659240722656
10     37     38.06023025512695
11     39     38.833797454833984
12     42     39.60737228393555
13     45     40.380943298339844
14     45     41.154518127441406
15     41     41.92808532714844
16     40     42.70166015625
17     39     43.47522735595703
18     42     44.24879837036133
19     44     45.02237319946289
20     47     45.79594039916992
21     49     46.56951904296875
22     50     47.34308624267578
23     49     48.116661071777344
24     46     48.890228271484375
25     48     49.717716217041016
26     50     50.6991081237793
27     53     51.775909423828125
28     55     52.898773193359375
29     54     54.052757263183594
30     53     55.230289459228516
```

#### Part I Problem (ii): Training data to a neural network and computing LSE
The calculated least square errors when using the first 20 data points training data, and then fitting the model to the test data (the last 10 data points) yielded the following results:
```
Least Square Error (Training Data): 5.029852557650884
Least Square Error (Test Data): 5.906001185996998
```

#### Part I Problem (iii): Training data to a neural network and computing LSE
The calculated least square errors when using the first 10 and last 10 data points as training data, and then fitting the model to the test data (the 10 held out middle data points) yielded the following results:
```
Least Square Error (Training Data): 4.670216516796063
Least Square Error (Test Data): 13.157855085386837
```

#### Part I Problem (iv): Comparison between linear regression vs. neural network models
The neural networks in both (ii) and (iii) have lower training and test data errors compared to the models in HW1. This suggests that the neural networks perform better in terms of fitting the data and generalizing to unseen test data.

Comparing the neural networks in (ii) and (iii), the neural network in (iii), which computes and fits a model using the first 10 and last 10 data points as training data, has a slightly lower training error but a higher test data error. This suggests that the neural network in (iii) may have overfit the training data to some extent, resulting in reduced generalization performance on unseen test data.

#### Part II Problem (i): PCA modes on the MNIST dataset
Computation of the first 20 PCA modes of the digit images and their respective variance ratios were determined to be as followed:
```
Shape of X_pca: (70000, 20)
Shape of pca_modes: (20, 784)

Explained variance ratio:
Component 1: 9.75%
Component 2: 7.16%
Component 3: 6.15%
Component 4: 5.40%
Component 5: 4.89%
Component 6: 4.31%
Component 7: 3.28%
Component 8: 2.89%
Component 9: 2.76%
Component 10: 2.34%
Component 11: 2.11%
Component 12: 2.04%
Component 13: 1.71%
Component 14: 1.69%
Component 15: 1.58%
Component 16: 1.49%
Component 17: 1.32%
Component 18: 1.28%
Component 19: 1.19%
Component 20: 1.15%
```

#### Part II Problem (ii): Neural networks, LSTM, SVM, and decision tree classifiers
The feed-forward neural network, LSTM classifier, SVM classifier, and decision tree classifiers yielded the following test accuracies: 
```
Test accuracy (feed-forward neural network): 0.9734285473823547
Test accuracy (LSTM classifier): 0.9865714311599731
Test accuracy (SVM classifier): 0.9764285714285714
Test accuracy (Decision Tree classifier): 0.8712142857142857
```

Overall, it is observed that the LSTM classifier achieved the highest test accuracy, indicating its superior performance in classifying the test data. However, the feed-forward neural network and SVM classifier also performed well, while the decision tree classifier had a comparatively lower accuracy.

## Summary and Conclusions:
The assignment presented a comprehensive exploration of digit classification using various machine learning models on the MNIST dataset. The task was divided into two parts, each focusing on different aspects of the classification problem.

Part I, concentrated on a specific dataset with numerical features and labels. Regression models were constructed using neural networks, and their performance was evaluated based on the mean squared error metric. The impact of different dataset splitting strategies on model performance and generalization was also examined.

In Part II, an array of models was implemented, including a feed-forward neural network, an LSTM classifier, an SVM classifier, and a decision tree classifier. The MNIST dataset was loaded, preprocessed, and normalized to ensure optimal performance. PCA was applied to reduce the dimensionality of the data, and the first 20 PCA modes were analyzed to understand their contribution to the overall variance. Each model was trained and evaluated, shedding light on their accuracy and efficiency in classifying digits.

Through this assignment, we gained valuable insights into the strengths and limitations of various machine learning techniques. SVM and decision trees exhibited solid performance, while deep learning models like feed-forward neural networks and LSTMs demonstrated their ability to capture complex patterns in the data. 

In conclusion, this assignment showcased the versatility and effectiveness of different machine learning models for digit classification. It underscored the significance of data preprocessing, model design, and evaluation techniques in achieving robust and accurate results. By leveraging the power of both traditional and deep learning approaches, we can enhance our understanding of digit classification tasks and pave the way for further advancements in the field of machine learning.
