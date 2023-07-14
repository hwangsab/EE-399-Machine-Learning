# EE-399: Introduction to Machine Learning
#### Application of Singular Value Decomposition Analysis
#### Sabrina Hwang

## Abstract:
This code performs a variety of tasks related to analyzing and classifying the MNIST dataset of handwritten digits. The MNIST dataset is a classic dataset widely used for machine learning tasks and contains 70,000 grayscale images of size 28x28 pixels, with each image corresponding to a digit from 0 to 9.

Overall, this code provides a useful demonstration of how to perform an analysis of the MNIST dataset using SVD, as well as how to train and test classifiers to identify individual digits in the dataset. It also showcases several different classifiers and provides a comparison of their performance. This code could be used as a starting point for further analysis and classification tasks involving the MNIST dataset or other similar datasets.

## Table of Contents:
* [Abstract](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#abstract)
* [Introduction and Overview](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#introduction-and-overview)
* [Theoretical Background](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#theoretical-background)
* [Algorithm Implementation and Development](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#algorithm-implementation-and-development)
  * [Code Description](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#code-description)
    * [Part I Problem 1: SVD Analysis of Digit Images with Random Sampling and Visualization](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-i-problem-1-svd-analysis-of-digit-images-with-random-sampling-and-visualization)
    * [Part I Problem 2: Singular Value Spectrum and Rank Estimation](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-i-problem-2-singular-value-spectrum-and-rank-estimation)
    * [Part I Problem 3: Interpretation of U, Σ, and V Matrices in SVD Analysis](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-i-problem-3-interpretation-of-u-%CF%83-and-v-matrices-in-svd-analysis)
    * [Part I Problem 4: Visualization of selected V-modes of PCA with 3D scatter plot](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-i-problem-4-visualization-of-selected-v-modes-of-pca-with-3d-scatter-plot)
    * [Part II Problem (a): Linear classification of two digits using LDA](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-ii-problem-a-linear-classification-of-two-digits-using-lda)
    * [Part II Problem (b): Linear classification of three digits using LDA](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-ii-problem-b-linear-classification-of-three-digits-using-lda)
    * [Part II Problem (c): Identifying the most difficult digit pairs to separate using LDA classifiers](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-ii-problem-c-identifying-the-most-difficult-digit-pairs-to-separate-using-lda-classifiers)
    * [Part II Problem (d): Identifying the most easy digit pairs to separate using LDA classifiers](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-ii-problem-d-identifying-the-most-easy-digit-pairs-to-separate-using-lda-classifiers)
    * [Part II Problem (e): Identifying most easy and difficult digit pairs using SVM and decision tree classifiers](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-ii-problem-e-identifying-most-easy-and-difficult-digit-pairs-using-svm-and-decision-tree-classifiers)
    * [Part II Problem (f): Comparing the performance between LDA, SVM, and decision tree classifiers](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-ii-problem-f-comparing-the-performance-between-lda-svm-and-decision-tree-classifiers)
* [Computational Results](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#computational-results)
  * [Usage](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#usage)
  * [Part I Problem 1: SVD Analysis of Digit Images with Random Sampling and Visualization](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-i-problem-1-svd-analysis-of-digit-images-with-random-sampling-and-visualization-1)
  * [Part I Problem 2: Singular Value Spectrum and Rank Estimation](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-i-problem-2-singular-value-spectrum-and-rank-estimation-1)
  * [Part I Problem 3: Interpretation of U, Σ, and V Matrices in SVD Analysis](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-i-problem-3-interpretation-of-u-%CF%83-and-v-matrices-in-svd-analysis-1)
  * [Part I Problem 4: Visualization of selected V-modes of PCA with 3D scatter plot](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-i-problem-4-visualization-of-selected-v-modes-of-pca-with-3d-scatter-plot-1)
  * [Part II Problem (a): Linear classification of two digits using LDA](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-ii-problem-a-linear-classification-of-two-digits-using-lda-1)
  * [Part II Problem (b): Linear classification of three digits using LDA](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-ii-problem-b-linear-classification-of-three-digits-using-lda-1)
  * [Part II Problem (c): Identifying the most difficult digit pairs to separate using LDA classifiers](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-ii-problem-c-identifying-the-most-difficult-digit-pairs-to-separate-using-lda-classifiers-1)
  * [Part II Problem (d): Identifying the most easy digit pairs to separate using LDA classifiers](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-ii-problem-d-identifying-the-most-easy-digit-pairs-to-separate-using-lda-classifiers-1)
  * [Part II Problem (e): Identifying most easy and difficult digit pairs using SVM and decision tree classifiers](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-ii-problem-e-identifying-most-easy-and-difficult-digit-pairs-using-svm-and-decision-tree-classifiers-1)
  * [Part II Problem (f): Comparing the performance between LDA, SVM, and decision tree classifiers](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#part-ii-problem-f-comparing-the-performance-between-lda-svm-and-decision-tree-classifiers-1)
* [Summary and Conclusion](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#summary-and-conclusions)

## Introduction and Overview:
The first part of the code performs an analysis of the MNIST dataset using Singular Value Decomposition (SVD). The data is first scaled to the range [0, 1], and a random sample of 4000 images is taken. The images are then reshaped into column vectors and the SVD is performed on the centered data. The first 10 singular values are printed and the digit images corresponding to the first 10 columns of the centered data matrix are plotted. The singular value spectrum is also plotted to determine the number of modes necessary for good image reconstruction.

The code then moves onto building a classifier to identify individual digits in the training set. Two digits are chosen and a Linear Discriminant Analysis (LDA) classifier is trained to classify the digits. The data is split into training and testing sets, and several different classifiers are trained and tested, including Logistic Regression, Decision Trees, and Support Vector Machines (SVMs). The accuracy of each classifier is evaluated and compared to the accuracy of the LDA classifier.

## Theoretical Background:
The code is performing binary classification of the MNIST dataset using three different classifiers: Linear Discriminant Analysis (LDA), Support Vector Machine (SVM), and Decision Tree. The objective of the code is to evaluate the performance of each classifier in separating pairs of digits from 0 to 9.

The MNIST dataset is a popular dataset in machine learning, consisting of 70,000 images of handwritten digits from 0 to 9, with 7,000 images for each digit. The dataset is split into a training set of 60,000 images and a test set of 10,000 images. Each image is 28x28 pixels, and each pixel has a grayscale value between 0 and 255. The objective is to classify each image into one of the ten digit classes.

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
The code is written in Python and uses the following overarching libraries:  
* `numpy` for numerical computing  
* `matplotlib` for data visualization  
* `math` for mathematical functions  
* `random` for random generation
* `scipy` for regression

The code also uses the following libraries for Part I:
* from `sklearn.datasets` import `fetch_openml`
* from `sklearn.decomposition` import `PCA`
* import `matplotlib.pyplot` as `plt`
* from `mpl_toolkits.mplot3d` import `Axes3D`
* from `scipy.io` import `loadmat`
* from `scipy.sparse.linalg` import `eigs`
* from `numpy` import `linalg`
  
And the following additional libraries for Part II:
* from `sklearn.discriminant_analysis` import `LinearDiscriminantAnalysis`
* from `sklearn.model_selection` import `train_test_split`
* from `sklearn.linear_model` import `LogisticRegression`
* from `sklearn.metrics` import `accuracy_score`
* from `sklearn.tree` import `DecisionTreeClassifier`
* from `sklearn.svm` import `SVC`

#### Part I Problem 1: SVD Analysis of Digit Images with Random Sampling and Visualization
The first problem involves performing an SVD (Singular Value Decomposition) analysis of the digit images. The images are first reshaped into column vectors. A random sample of 4000 images is extracted for this problem. The SVD of the centered data is then computed using the numpy `linalg.svd()` function, and the first 10 singular values and their corresponding digit images are printed.

Computation of SVD from the centered data is dependent on the following line of code
```
U, S, Vt = np.linalg.svd(X_sample, full_matrices=False)
```
#### Part I Problem 2: Singular Value Spectrum and Rank Estimation
The second problem involves finding the number of modes (rank r of the digit space) necessary for good image reconstruction by analyzing the singular value spectrum. The SVD is performed on the full dataset, and the index of the first singular value that explains at least 90% of the total variance is found. The proportion of total variance explained by each singular value is then computed and plotted to show the singular value spectrum.

The index of the first singular value that explains at least 90% of the total variance was calculated using ```r = np.sum(S > ((1.00 - 0.9) * s[0]))```
Whereas the computation of the proportion of total variance was explained by `var_exp = (S**2)`

#### Part I Problem 3: Interpretation of U, Σ, and V Matrices in SVD Analysis
The third problem asks for the interpretation of the U, Σ, and V matrices in SVD. There is no explicit code to answer this problem.

#### Part I Problem 4: Visualization of selected V-modes of PCA with 3D scatter plot
The fourth problem involves projecting the images onto three selected V-modes (columns) colored by their digit label on a 3D plot. For this problem, the MNIST data is again loaded, and PCA (Principal Component Analysis) is performed on the data using the `PCA` function from `sklearn.decomposition`. The second, third, and fifth principal components are selected, and the 3D scatter plot is created using the `mpl_toolkits.mplot3d` module.

PCA was performed on the data by using the lines
```
pca = PCA(n_components=784)
X_pca = pca.fit_transform(X)
```
And then the 2nd, 3rd, and 5th principal components were selected to create a scatter plot using the following code:
```
# Select the 2nd, 3rd, and 5th principal components
v_modes = [1, 2, 4]  # Note that we use 1-based indexing here
X_pca_selected = X_pca[:, v_modes]

# Create the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_selected[:, 0], X_pca_selected[:, 1], X_pca_selected[:, 2], c=y.astype(int), s=1)
```

#### Part II Problem (a): Linear classification of two digits using LDA
This problem involves the selection of two digits (3 and 8) from the dataset and tries to build a linear classifier to classify them. It first selects only the data samples for these two digits and applies LDA to reduce the dimensionality of the data. It then trains a logistic regression classifier on the transformed data and evaluates its accuracy on a test set.

The selection of the two digits (in this case, 3 and 8) were used by the following lines of code:
```
digit1 = 3
digit2 = 8
X = X[(y == digit1) | (y == digit2)]
y = y[(y == digit1) | (y == digit2)]
y[y == digit1] = 0
y[y == digit2] = 1
```

And the subsequent application of LDA, training of a logistic regression classifier and applying the LDA to the test data was computed using the following code:
```
# Apply LDA to reduce the dimensionality of the data
lda = LinearDiscriminantAnalysis()
X_lda_train = lda.fit_transform(X_train, y_train)

# Train a logistic regression classifier on the LDA-transformed data
clf = LogisticRegression(random_state=42)
clf.fit(X_lda_train, y_train)

# Apply LDA to the test data and make predictions
X_lda_test = lda.transform(X_test)
y_pred = clf.predict(X_lda_test)
```

#### Part II Problem (b): Linear classification of three digits using LDA
Problem (b) is an extension of problem (a), only now the code selects three digits (3, 7, and 8) from the dataset and tries to build a linear classifier to classify them. It follows the same process as in section (a) to prepare the data and train a classifier.

The selection of the three digits (in this case, 3, 7 and 8) were used by the following lines of code:
```
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
```

And the subsequent application of LDA, training of a logistic regression classifier and applying the LDA to the test data was computed using the same steps as prior in problem (a). 

#### Part II Problem (c): Identifying the most difficult digit pairs to separate using LDA classifiers
This problem compares all pairs of digits in the dataset to determine which pair is most difficult to separate. It calculates the accuracy of the LDA classifier on the test set for each pair of digits and stores the results in a dictionary.

This problem was implemented by creating a loop that would implement through all existing pairs of digits within the data set.
```
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
```

Then, the determination of which pair was most difficult was computed by finding the pair that had the lowest accuracy value.
```
most_difficult_pair_lda = min(accuracy_dict_lda, key=accuracy_dict_lda.get)
```

#### Part II Problem (d): Identifying the most easy digit pairs to separate using LDA classifiers
This problem asks to determine the two digits that are the easiest to separate, the code compares all pairs of digits and stores their accuracy using LDA for dimensionality reduction and Logistic Regression for classification. The digit pair with the highest accuracy is considered the easiest to separate.

The process and implementation for this problem is the same as the implementation for problem (c), only this time the determination of which pair was most easy was computed by finding the pair that had the highest accuracy value:

```
most_easy_pair_lda = max(accuracy_dict_lda, key=accuracy_dict_lda.get)
```

#### Part II Problem (e): Identifying most easy and difficult digit pairs using SVM and decision tree classifiers
Similarly to problem (d) and problem (e), this problem asks for computation of the two digits easiest to separate, as well as the two digits most difficult to separate, using SVM and decision tree classifiers. 

As such, the implementation of this code is the same as the code for problem (d) and problem (e), only different on the type of classifier being trained. 
For example, whereas the SVM classifier uses the following line of code: 
```
clf = SVC(kernel='linear', random_state=42)
clf.fit(X_lda_train, y_train)
```

The decision tree classifier will use this line instead:
```
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
```

#### Part II Problem (f): Comparing the performance between LDA, SVM, and decision tree classifiers
This problem asks to compare the performance between LDA, SVM, and decision trees on the hardest and easiest pair of digits to separate. There is also no explicit code to answer this problem.

## Computational Results:

### Usage
To run the code, simply run the Python file `EE399 HW3.py` in any Python environment. The output will be 
printed to the console and displayed in a pop-up window.

#### Part I Problem 1: SVD Analysis of Digit Images with Random Sampling and Visualization
The resultant 10 singular values of the SVD centered data were determined to be:
```
Random sample shape:  (784, 4000)
First 10 singular values:
     U shape:   (784, 784)
     S shape:   (784,)
     Vt shape:  (784, 4000)
```
In addition, the printed digit images corresponding to the first 10 columns of the centered data matrix are displayed: 
![download](https://user-images.githubusercontent.com/125385468/234189618-7bb0396d-7dab-45a5-a76c-0d3ca4c840b8.png)

#### Part I Problem 2: Singular Value Spectrum and Rank Estimation
The plotted singular value spectrum is as follows:  
![download](https://user-images.githubusercontent.com/125385468/234189438-3ea0b072-3c78-437a-85b8-5523b0c17f5c.png)

From the code, we were able to compute that the rank of digit space for 90% of the variance explained was 28. 

#### Part I Problem 3: Interpretation of U, Σ, and V Matrices in SVD Analysis
Our interpretation of the different matrices were as followed:
* Interpretation of U matrix: principal directions of the data The columns of U are the principal directions (eigenvectors) of the covariance matrix of the data The i-th column of U is the direction of greatest variance in the data projected onto the i-th principal axis U is an orthogonal matrix, so the columns are unit vectors and are mutually orthogonal
* Interpretation of s vector: singular values The singular values are the square roots of the eigenvalues of the covariance matrix of the data The singular values are non-negative and in non-increasing order They represent the amount of variance in the data that is explained by each principal direction
* Interpretation of V matrix: principal components of the data The rows of V are the principal components of the data The i-th row of V is the weight (or contribution) of each feature (pixel) to the i-th principal component V is also an orthogonal matrix, so the rows are unit vectors and are mutually orthogonal

#### Part I Problem 4: Visualization of selected V-modes of PCA with 3D scatter plot
The three selected V-modes were visualized into the following plot: 
![download](https://user-images.githubusercontent.com/125385468/234191754-e5cc932f-7281-4024-8174-708ba131d0e6.png)

#### Part II Problem (a): Linear classification of two digits using LDA
The linear classifier for 2 selected digits was built around the following parameters:
```
Number of samples for digit 3: 7141
Number of samples for digit 8: 6825

Number of training samples: 11172
Number of test samples: 2794

Coefficients of the linear boundary: [2.70133468]
Point of intersection: [-0.0986239]
Accuracy: 0.96
```

#### Part II Problem (b): Linear classification of three digits using LDA
The linear classifier for 3 selected digits was built around the following parameters:
```
Number of samples for digit 3: 7141
Number of samples for digit 7: 0
Number of samples for digit 8: 0

Number of training samples: 17007
Number of test samples: 4252

Coefficients of the linear boundary: [[ 0.58481592  1.30106924]
 [-1.60590181 -0.08605543]
 [ 1.02108589 -1.21501381]]
Point of intersection: [ 0.49113982 -0.47289057 -0.01824925]
Accuracy: 0.95
```

#### Part II Problem (c): Identifying the most difficult digit pairs to separate using LDA classifiers
Using the LDA classifier to determine which digit pairs was most easy to separte it was concluded that the most difficult pair of digits to separate with LDA is (3, 5) with an accuracy of 0.95.

#### Part II Problem (d): Identifying the most easy digit pairs to separate using LDA classifiers
Using the LDA classifier to determine which digit pairs was most easy to separte it was concluded that the easiest pair of digits to separate with LDA is (6, 7) with an accuracy of 1.00.

#### Part II Problem (e): Identifying most easy and difficult digit pairs using SVM and decision tree classifiers
The next two classifiers that were tested yielded the following conclusions on which digit pairs were most difficult and most easy to separate: 
```
Support vector machine computation:
The most difficult pair of digits to separate is (3, 5) with an accuracy of 0.95.
The most easy pair of digits to separate is (0, 1) with an accuracy of 1.00.
  
Decision tree computation:
The most difficult pair of digits to separate is (2, 3) with an accuracy of 0.95.
The most easy pair of digits to separate is (0, 1) with an accuracy of 1.00.
```

#### Part II Problem (f): Comparing the performance between LDA, SVM, and decision tree classifiers
From what could be observed, it appears that the most difficult digit to distinguish is 3, which is observed to have the lowest accuracy of distinguishment with digit 5 for both the LDA and SVM classifiers, and with digit 2 for the decision tree classifier.

On the other hand, the most easy digit to distinguish appears to be 0 and 1, which are distinguished with the SVM and the decision tree model with a full accuracy of 100%.

Overall, the SVM classifier shares its performance with the LDA classifier for determining the most difficult pair of digits to separate, but shares its performance with the decision tree classifier for determining the most easy pair of digits to separate. The LDA classifier and decision trees classifier share the least in common, besides the overarching idea that the digit 3 is the most difficult digit to distinguish.

## Summary and Conclusions:
Overall, this assignment provided a comprehensive analysis of the MNIST dataset using various classification techniques, and evaluated their performance on different pairs of digits. It demonstrated the importance of feature selection and the power of different classifiers for identifying patterns in data.
