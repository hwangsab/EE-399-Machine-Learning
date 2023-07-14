#!/usr/bin/env python
# coding: utf-8

# # EE 399 Introduction to Machine Learning: HW 4 Submission
# ### Sabrina Hwang

# In[18]:


# GitHub HW4: https://github.com/hwangsab/EE-399-HW4


# ## Part 1: Fitting data to a feed forward neural network

# In[10]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# In[11]:


# Define the dataset
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 
              40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])


# ### Problem (i)
# Fit the data to a three layer feed forward neural network

# In[19]:


# Normalize the input data
X_normalized = (X - np.mean(X)) / np.std(X)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_normalized, Y, epochs=1000, verbose=0)

# Predict using the trained model
X_test = np.arange(0, 31)
X_test_normalized = (X_test - np.mean(X)) / np.std(X)
predictions = model.predict(X_test_normalized)

# Print the predicted and actual values
print("X\tY\tPredicted Y")
print("--------------------")
for x, y, pred in zip(X, Y, predictions.flatten()):
    print(f"{x}\t{y}\t{pred}")


# ### Problem (ii)
# Using the first 20 data points as training data, fit the neural network. Compute the
# least-square error for each of these over the training points. Then compute the least
# square error of these models on the test data which are the remaining 10 data points.

# In[22]:


# Normalize the input data
X_normalized = (X - np.mean(X)) / np.std(X)

# Split the data into training and test sets
X_train = X_normalized[:20]
Y_train = Y[:20]
X_test = X_normalized[20:]
Y_test = Y[20:]

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model on the training data
model.fit(X_train, Y_train, epochs=1000, verbose=0)

# Predict using the trained model on training and test data
predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

# Compute the least square error for training and test data
lsq_error_train = np.mean((predictions_train.flatten() - Y_train) ** 2)
lsq_error_test = np.mean((predictions_test.flatten() - Y_test) ** 2)

# Print the least square errors
print("Least Square Error (Training Data):", lsq_error_train)
print("Least Square Error (Test Data):", lsq_error_test)


# ### Problem (iii)
# Repeat (ii) but use the first 10 and last 10 data points as training data. Then fit the
# model to the test data (which are the 10 held out middle data points). Compare these
# results to (ii)

# In[24]:


# Normalize the input data
X_normalized = (X - np.mean(X)) / np.std(X)

# Split the data into training and test sets
X_train = np.concatenate((X_normalized[:10], X_normalized[-10:]))
Y_train = np.concatenate((Y[:10], Y[-10:]))
X_test = X_normalized[10:20]
Y_test = Y[10:20]

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model on the training data
model.fit(X_train, Y_train, epochs=1000, verbose=0)

# Predict using the trained model on the training and test data
predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

# Compute the least square error for training and test data
lsq_error_train = np.mean((predictions_train.flatten() - Y_train) ** 2)
lsq_error_test = np.mean((predictions_test.flatten() - Y_test) ** 2)

# Print the least square errors
print("Least Square Error (Training Data):", lsq_error_train)
print("Least Square Error (Test Data):", lsq_error_test)


# ### Problem (iv)
# Compare the models fit in homework one to the neural networks in (ii) and (iii)

# The neural networks in both (ii) and (iii) have lower training and test data errors compared to the models in HW1. This suggests that the neural networks perform better in terms of fitting the data and generalizing to unseen test data.
# 
# Comparing the neural networks in (ii) and (iii), the neural network in (iii), which computes and fits a model using the first 10 and last 10 data points as training data, has a slightly lower training error but a higher test data error. This suggests that the neural network in (iii) may have overfit the training data to some extent, resulting in reduced generalization performance on unseen test data.

# ## Part 2: Training a feedforward neural network on the MNIST data set

# In[1]:


from sklearn.datasets import fetch_openml
from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import tensorflow as tf


# In[2]:


# Load the MNIST data
mnist = fetch_openml('mnist_784', parser='auto')

# Extract the features and labels
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# Normalize the features
X /= 255.0


# ### Question (i) 
# Compute the first 20 PCA modes of the digit images.

# In[3]:


# Compute PCA with 20 components
n_components = 20
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Print the shape of X_pca (optional)
print("Shape of X_pca:", X_pca.shape)

# Access the first 20 PCA modes
pca_modes = pca.components_

# Print the shape of pca_modes (optional)
print("Shape of pca_modes:", pca_modes.shape)
print()

# Access the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Format the explained variance ratio for better readability
formatted_variance_ratio = ["{:.2%}".format(ratio) for ratio in explained_variance_ratio]

# Print the formatted explained variance ratio for the first 20 components
print("Explained variance ratio:")
for i, ratio in enumerate(formatted_variance_ratio[:n_components]):
    print("Component {}: {}".format(i+1, ratio))


# ### Question (ii) 
# Build a feed-forward neural network to classify the digits. Compare the results of
# the neural network against LSTM, SVM (support vector machines) and decision tree
# classifiers.

# In[4]:


# Feed-forward neural network
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the architecture of the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test loss (feed-forward neural network):", test_loss)
print("Test accuracy (feed-forward neural network):", test_accuracy)


# In[5]:


# LSTM classifier
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data for LSTM
X_train = X_train.values.reshape(-1, 28, 28)
X_test = X_test.values.reshape(-1, 28, 28)

# Define the architecture of the LSTM classifier
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy (LSTM classifier):", test_accuracy)


# In[6]:


# SVM Classifier
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the SVM classifier
classifier = SVC()

# Train the classifier
classifier.fit(X_train, y_train)

# Evaluate the classifier on the test set
test_accuracy = classifier.score(X_test, y_test)
print("Test accuracy (SVM classifier):", test_accuracy)


# In[7]:


# Decision Tree Classifier
# Convert the labels to one-hot encoded vectors
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the decision tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Evaluate the classifier on the test set
test_accuracy = classifier.score(X_test, y_test)
print("Test accuracy (Decision Tree classifier):", test_accuracy)


# The LSTM classifier achieved the highest test accuracy of 0.9865714311599731. This indicates that the LSTM classifier performed the best among the models evaluated, showing the highest level of accuracy in predicting the correct class labels for the test data.
# 
# The feed-forward neural network achieved a test accuracy slightly lower than the LSTM classifier. However, it still performed well and demonstrated a high level of accuracy in classifying the test data.
# 
# The SVM classifier performed slightly better than the feed-forward neural network but slightly lower than the LSTM classifier. SVMs are known for their effectiveness in handling complex classification tasks, and this accuracy suggests that the SVM classifier was able to accurately classify the test data.
# 
# The decision tree classifier achieved the lowest test accuracy. Although it achieved a relatively lower accuracy compared to the other models, it still provides some level of prediction capability. Decision trees might be simpler models compared to neural networks and SVMs, and they might struggle with complex datasets, resulting in a lower accuracy.
# 
# In summary, the LSTM classifier achieved the highest test accuracy, indicating its superior performance in classifying the test data. However, the feed-forward neural network and SVM classifier also performed well, while the decision tree classifier had a comparatively lower accuracy.

# In[ ]:




