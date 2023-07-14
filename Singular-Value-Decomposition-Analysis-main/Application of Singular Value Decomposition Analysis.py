#!/usr/bin/env python
# coding: utf-8

# # EE 399 Introduction to Machine Learning: HW 3 Submission
# ### Sabrina Hwang

# In[45]:


# GitHub HW3: https://github.com/hwangsab/EE-399-HW3


# ## Part 1: Performing an analysis of the MNIST data set.

# In[46]:


import numpy as np
import matplotlib.pyplot as plt
import math as math
import random
import scipy

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.io import loadmat
from scipy.sparse.linalg import eigs
from numpy import linalg


# In[47]:


# Load the MNIST data
mnist = fetch_openml('mnist_784', parser='auto')
X = mnist.data / 255.0  # Scale the data to [0, 1]
y = mnist.target


# ### Problem 1
# Do an SVD analysis of the digit images. You will need to reshape each image into a column vector
# and each column of your data matrix is a different image.

# In[48]:


# Reshape the images into column vectors
X = np.array(X).T

# Extracting random sample for sack of the problem
rand = random.sample(range(X.shape[1]), 4000)
X_sample = X[:, rand]
Y_sample = y[rand]
print("Random sample shape: ", X_sample.shape)

# Compute the SVD of the centered data
U, S, Vt = np.linalg.svd(X_sample, full_matrices=False)

# Print the first 10 singular values
print('First 10 singular values:')
print("     U shape:  ", U.shape) 
print("     S shape:  ", S.shape)
print("     Vt shape: ", Vt.shape)

# Print the digit images corresponding to the first 10 columns of the centered data matrix
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(12, 15))
for i, ax in enumerate(axs.flat):
    ax.imshow(X_sample[:, i].reshape((28, 28)), cmap='gray')
    ax.set_title('Digit {}'.format(i))
plt.tight_layout()
plt.show()


# ### Problem 2
# What does the singular value spectrum look like and how many modes are necessary for good image reconstruction? (i.e. what is the rank r of the digit space?)

# In[49]:


# Extract the images and labels
X, y = mnist.data, mnist.target

# Perform SVD on the data
U, S, Vt = np.linalg.svd(X_sample, full_matrices=False)

# Find the index of the first singular value that explains at least 90% of the total variance
r = np.sum(S > ((1.00 - 0.9) * s[0]))
print(f"Rank of digit space for 90% of variance explained: {r}")

# Compute the proportion of total variance explained by each singular value
var_exp = (S**2)

# Plot the singular value spectrum
plt.plot(var_exp)
plt.title('Singular Value Spectrum')
plt.xlabel('Singular Value Index')
plt.ylabel('Proportion of Total Variance')
plt.show()


# ### Problem 3
# What is the interpretation of the U, Σ, and V matrices?

# Interpretation of U matrix: principal directions of the data
# The columns of U are the principal directions (eigenvectors) of the covariance matrix of the data
# The i-th column of U is the direction of greatest variance in the data projected onto the i-th principal axis
# U is an orthogonal matrix, so the columns are unit vectors and are mutually orthogonal
# 
# Interpretation of s vector: singular values
# The singular values are the square roots of the eigenvalues of the covariance matrix of the data
# The singular values are non-negative and in non-increasing order
# They represent the amount of variance in the data that is explained by each principal direction
# 
# Interpretation of V matrix: principal components of the data
# The rows of V are the principal components of the data
# The i-th row of V is the weight (or contribution) of each feature (pixel) to the i-th principal component
# V is also an orthogonal matrix, so the rows are unit vectors and are mutually orthogonal

# ### Problem 4
# On a 3D plot, project onto three selected V-modes (columns) colored by their digit label. For example, columns 2,3, and 5.

# In[50]:


# Load the MNIST data
X, y = mnist.data, mnist.target
X = X / 255.0 # Scale the pixel values to [0, 1]

# Perform PCA on the data
pca = PCA(n_components=784)
X_pca = pca.fit_transform(X)

# Select the 2nd, 3rd, and 5th principal components
v_modes = [1, 2, 4]  # Note that we use 1-based indexing here
X_pca_selected = X_pca[:, v_modes]

# Create the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_selected[:, 0], X_pca_selected[:, 1], X_pca_selected[:, 2], c=y.astype(int), s=1)

# Set the labels and limits of the plot
ax.set_xlabel('2nd Principal Component')
ax.set_ylabel('3rd Principal Component')
ax.set_zlabel('5th Principal Component')

# Add a title to the plot
plt.title('MNIST Data Visualization Using 2nd, 3rd, and 5th Principal Components')

plt.show()


# ## Part 2: Building a classifier to identify individual digits in the training set

# In[37]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[38]:


# Load the MNIST data
mnist = fetch_openml('mnist_784', parser='auto')


# ### Question (a) 
# Pick two digits. See if you can build a linear classifier (LDA) that can reasonably identify/classify them. 

# In[39]:


X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# Select only two digits
digit1 = 3
digit2 = 8
X = X[(y == digit1) | (y == digit2)]
y = y[(y == digit1) | (y == digit2)]
y[y == digit1] = 0
y[y == digit2] = 1

# Display the number of samples for each digit
print(f"Number of samples for digit {digit1}: {len(y[y==0])}")
print(f"Number of samples for digit {digit2}: {len(y[y==1])}\n")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the split between the train and test digits
print(f"Number of training samples: {len(y_train)}")
print(f"Number of test samples: {len(y_test)}\n")

# Apply LDA to reduce the dimensionality of the data
lda = LinearDiscriminantAnalysis()
X_lda_train = lda.fit_transform(X_train, y_train)

# Train a logistic regression classifier on the LDA-transformed data
clf = LogisticRegression(random_state=42)
clf.fit(X_lda_train, y_train)

# Display the first 10 coefficients of the linear boundary
print(f"Coefficients of the linear boundary: {clf.coef_[:,:10]}")

# Display the point of intersection of the lines
print(f"Point of intersection: {clf.intercept_/clf.coef_[0,-1]}")

# Apply LDA to the test data and make predictions
X_lda_test = lda.transform(X_test)
y_pred = clf.predict(X_lda_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# ### Question (b) 
# Pick three digits. Try to build a linear classifier to identify these three now.

# In[40]:


# Select three digits
digit1 = 3
digit2 = 7
digit3 = 8

X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

X = X[(y == digit1) | (y == digit2) | (y == digit3)]
y = y[(y == digit1) | (y == digit2) | (y == digit3)]
y[y == digit1] = 0
y[y == digit2] = 1
y[y == digit3] = 2

# Display the number of samples for each digit
for digit in [digit1, digit2, digit3]:
    print(f"Number of samples for digit {digit}: {len(y[y == (digit - digit1)])}")

print()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the split between the train and test digits
print(f"Number of training samples: {len(y_train)}")
print(f"Number of test samples: {len(y_test)}\n")

# Apply LDA to reduce the dimensionality of the data
lda = LinearDiscriminantAnalysis()
X_lda_train = lda.fit_transform(X_train, y_train)

# Train a logistic regression classifier on the LDA-transformed data
clf = LogisticRegression(random_state=42)
clf.fit(X_lda_train, y_train)

# Display the first 10 coefficients of the linear boundary
print(f"Coefficients of the linear boundary: {clf.coef_[:,:10]}")

# Display the point of intersection of the lines
print(f"Point of intersection: {clf.intercept_ / clf.coef_[0,-1]}")

# Apply LDA to the test data and make predictions
X_lda_test = lda.transform(X_test)
y_pred = clf.predict(X_lda_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# ### Question (c)
# Which two digits in the data set appear to be the most difficult to separate? Quantify the accuracy of the separation with LDA on the test data.

# In[41]:


# Convert data and target to appropriate types
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# Identify all pairs of digits to compare
digit_pairs = [(i, j) for i in range(10) for j in range(i+1, 10)]

# Initialize dictionaries to store accuracy for each digit pair
accuracy_dict_lda = {}

# Loop through all pairs of digits
for digit1, digit2 in digit_pairs:
    
    # Select only two digits
    X_pair = X[(y == digit1) | (y == digit2)]
    y_pair = y[(y == digit1) | (y == digit2)]
    y_pair[y_pair == digit1] = 0
    y_pair[y_pair == digit2] = 1
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pair, y_pair, test_size=0.2, random_state=42)
    
    # Apply LDA to reduce the dimensionality of the data
    lda = LinearDiscriminantAnalysis()
    X_lda_train = lda.fit_transform(X_train, y_train)
    X_lda_test = lda.transform(X_test)
    
    # Train a logistic regression classifier on the LDA-transformed data
    clf = LogisticRegression(random_state=42)
    clf.fit(X_lda_train, y_train)
    
    # Make predictions on the test data
    y_pred = clf.predict(X_lda_test)
    
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the accuracy for this digit pair in the dictionary
    accuracy_dict_lda[(digit1, digit2)] = accuracy
    
# Find the pair of digits with the lowest accuracy
most_difficult_pair_lda = min(accuracy_dict_lda, key=accuracy_dict_lda.get)

# Print the most difficult and easiest pairs of digits and their accuracies
print(f"The most difficult pair of digits to separate with LDA is {most_difficult_pair_lda} with an accuracy of {accuracy_dict_lda[most_difficult_pair_lda]:.2f}.")


# ### Question (d)
# Which two digits in the data set are most easy to separate? Quantify the accuracy of the separation with LDA on the test data.

# In[42]:


# Convert data and target to appropriate types
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# Identify all pairs of digits to compare
digit_pairs = [(i, j) for i in range(10) for j in range(i+1, 10)]

# Initialize dictionaries to store accuracy for each digit pair
accuracy_dict_lda = {}

# Loop through all pairs of digits
for digit1, digit2 in digit_pairs:
    
    # Select only two digits
    X_pair = X[(y == digit1) | (y == digit2)]
    y_pair = y[(y == digit1) | (y == digit2)]
    y_pair[y_pair == digit1] = 0
    y_pair[y_pair == digit2] = 1
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pair, y_pair, test_size=0.2, random_state=42)
    
    # Apply LDA to reduce the dimensionality of the data
    lda = LinearDiscriminantAnalysis()
    X_lda_train = lda.fit_transform(X_train, y_train)
    X_lda_test = lda.transform(X_test)
    
    # Train a logistic regression classifier on the LDA-transformed data
    clf = LogisticRegression(random_state=42)
    clf.fit(X_lda_train, y_train)
    
    # Make predictions on the test data
    y_pred = clf.predict(X_lda_test)
    
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the accuracy for this digit pair in the dictionary
    accuracy_dict_lda[(digit1, digit2)] = accuracy
    
# Find the pair of digits with the lowest and highest accuracy
most_easy_pair_lda = max(accuracy_dict_lda, key=accuracy_dict_lda.get)

# Print the most difficult and easiest pairs of digits and their accuracies
print(f"The easiest pair of digits to separate with LDA is {most_easy_pair_lda} with an accuracy of {accuracy_dict_lda[most_easy_pair_lda]:.2f}.")


# ### Question (e) 
# SVM (support vector machines) and decision tree classifiers were the state-of-the-art until about 2014. How well do these separate between all ten digits? (see code below to get started).

# In[43]:


# Support vector machine computation
# Convert data and target to appropriate types
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# Identify all pairs of digits to compare
digit_pairs = [(i, j) for i in range(10) for j in range(i+1, 10)]

# Initialize dictionary to store accuracy for each digit pair
accuracy_dict = {}

# Loop through all pairs of digits
for digit1, digit2 in digit_pairs:
    
    # Select only two digits
    X_pair = X[(y == digit1) | (y == digit2)]
    y_pair = y[(y == digit1) | (y == digit2)]
    y_pair[y_pair == digit1] = 0
    y_pair[y_pair == digit2] = 1
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pair, y_pair, test_size=0.2, random_state=42)
    
    # Apply LDA to reduce the dimensionality of the data
    lda = LinearDiscriminantAnalysis()
    X_lda_train = lda.fit_transform(X_train, y_train)
    X_lda_test = lda.transform(X_test)
    
    # Train an SVM classifier on the LDA-transformed data
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_lda_train, y_train)
    
    # Make predictions on the test data
    y_pred = clf.predict(X_lda_test)
    
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the accuracy for this digit pair in the dictionary
    accuracy_dict[(digit1, digit2)] = accuracy
    
# Find the pair of digits with the lowest accuracy
most_difficult_pair = min(accuracy_dict, key=accuracy_dict.get)

# Print the most difficult pair of digits and its accuracy
print(f"The most difficult pair of digits to separate is {most_difficult_pair} with an accuracy of {accuracy_dict[most_difficult_pair]:.2f}.")

# Find the pair of digits with the highest accuracy
most_easy_pair = max(accuracy_dict, key=accuracy_dict.get)

# Print the most easy pair of digits and its accuracy
print(f"The most easy pair of digits to separate is {most_easy_pair} with an accuracy of {accuracy_dict[most_easy_pair]:.2f}.")


# In[44]:


# Decision tree computation
# Convert data and target to appropriate types
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# Identify all pairs of digits to compare
digit_pairs = [(i, j) for i in range(10) for j in range(i+1, 10)]

# Initialize dictionary to store accuracy for each digit pair
accuracy_dict = {}

# Loop through all pairs of digits
for digit1, digit2 in digit_pairs:
    
    # Select only two digits
    X_pair = X[(y == digit1) | (y == digit2)]
    y_pair = y[(y == digit1) | (y == digit2)]
    y_pair[y_pair == digit1] = 0
    y_pair[y_pair == digit2] = 1
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pair, y_pair, test_size=0.2, random_state=42)
    
    # Train a decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the accuracy for this digit pair in the dictionary
    accuracy_dict[(digit1, digit2)] = accuracy
    
# Find the pair of digits with the lowest accuracy
most_difficult_pair = min(accuracy_dict, key=accuracy_dict.get)

# Find the pair of digits with the highest accuracy
most_easy_pair = max(accuracy_dict, key=accuracy_dict.get)

# Print the most difficult and easiest pair of digits and their accuracies
print(f"The most difficult pair of digits to separate is {most_difficult_pair} with an accuracy of {accuracy_dict[most_difficult_pair]:.2f}.")
print(f"The most easy pair of digits to separate is {most_easy_pair} with an accuracy of {accuracy_dict[most_easy_pair]:.2f}.")


# ### Compare the performance between LDA, SVM and decision trees on the hardest and easiest pair of digits to separate (from above).

# From what could be observed, it appears that the most difficult digit to distinguish is 3, which is observed to have the lowest accuracy of distinguishment with digit 5 for both the LDA and SVM classifiers, and with digit 2 for the decision tree classifier. 
# 
# On the other hand, the most easy digit to distinguish appears to be 0 and 1, which are distinguished with the SVM and the decision tree model with a full accuracy of 100%. 
# 
# Overall, the SVM classifier shares its performance with the LDA classifier for determining the most difficult pair of digits to separate, but shares its performance with the decision tree classifier for determining the most easy pair of digits to separate. The LDA classifier and decision trees classifier share the least in common, besides the overarching idea that the digit 3 is the most difficult digit to distinguish. 

# In[ ]:




