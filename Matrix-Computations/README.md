# EE-399: Introduction to Machine Learning
#### Applications of Matrix Computations
#### Sabrina Hwang

## Abstract:
This code uses a file that has a total of 39 different faces with about 65 lighting scenes for each 
face (2414 faces in all) in the form of a matrix. The individual images of the columns in the matrix 
$X$, where each image has been downsampled to 32x32 pixels and coverted into greyscale with values 
between 0 and 1. 

The accompanying Python code performs correlation matrix computations, and computes the correlation 
between the set of images provided within the matrix. In addition, the code makes additional 
computations for SVD, eigenvector comparisons, and percentage of variance. 

## Table of Contents:
* [Abstract](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#abstract)
* [Introduction and Overview](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#introduction-and-overview)
* [Theoretical Background](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#theoretical-background)
* [Algorithm Implementation and Development](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#algorithm-implementation-and-development)
  * [Code Description](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#code-description)
    * [Problem (a): Computing Correlation Matrix using Dot Product](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-a-computing-correlation-matrix-using-dot-product)
    * [Problem (b): Identifying Highly Correlated and Uncorrelated Images](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-b-identifying-highly-correlated-and-uncorrelated-images)
    * [Problem (c): Computing Correlation Matrix for Subset of Images](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-c-computing-correlation-matrix-for-subset-of-images)
    * [Problem (d): Finding the First Six Eigenvectors of $Y=XX^T$](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-d-finding-the-first-six-eigenvectors-of-y--xxt)
    * [Problem (e): Finding the First Six Principal Component Directions using SVD](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-e-finding-the-first-six-principal-component-directions-using-svd)
    * [Problem (f): Comparing First Eigenvector and First SVD Mode](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-f-comparing-first-eigenvector-and-first-svd-mode)
    * [Problem (g): Computing Variance Captured by Each of the First 6 SVD Modes and Plotting Them](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-g-computing-variance-captured-by-each-of-the-first-6-svd-modes-and-plotting-them)
* [Computational Results](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#computational-results)
  * [Usage](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#usage)
  * [Problem (a): Computing Correlation Matrix using Dot Product](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-a-computing-correlation-matrix-using-dot-product-1)
  * [Problem (b): Identifying Highly Correlated and Uncorrelated Images](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-a-computing-correlation-matrix-using-dot-product-1)
  * [Problem (c): Computing Correlation Matrix for Subset of Images](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-c-computing-correlation-matrix-for-subset-of-images-1)
  * [Problem (d): Finding the First Six Eigenvectors of $Y=XX^T$](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-d-finding-the-first-six-eigenvectors-of-yxxt)
  * [Problem (e): Finding the First Six Principal Component Directions using SVD](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-e-finding-the-first-six-principal-component-directions-using-svd-1)
  * [Problem (f): Comparing First Eigenvector and First SVD Mode](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-f-comparing-first-eigenvector-and-first-svd-mode-1)
  * [Problem (g): Computing Variance Captured by Each of the First 6 SVD Modes and Plotting Them](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-g-computing-variance-captured-by-each-of-the-first-6-svd-modes-and-plotting-them-1)
* [Summary and Conclusion](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#summary-and-conclusions)

## Introduction and Overview:
In this homework assignment, we will explore a dataset of 39 different faces, each with about 65 
different lighting scenes, for a total of 2414 images. The images are downsampled to 32x32 pixels 
and converted into grayscale with values ranging from 0 to 1. The dataset is stored in a matrix $X$ 
of size 1024x2414, where each column corresponds to an image.

The first task (a) is to compute a 100x100 correlation matrix $C$, where each element represents the 
correlation between two images. The correlation is computed as the dot product between the two 
images' vectors. We will plot the correlation matrix using the pcolor function.

From the correlation matrix, we will identify the two images that are most highly correlated and the 
two images that are least correlated in part (b). We will plot these faces to visually compare the 
similarities and differences between them.

Next, we will repeat the correlation matrix computation in (c), but this time we will compute a 10x10 
matrix and plot it. This will allow us to compare the correlation between images in a smaller subset 
of the dataset.

In parts (d) and (e), we will use different methods to find the first six eigenvectors and principal 
component directions, respectively, of the matrix $X$. We will then compare the first eigenvector 
found using both methods and compute the norm of the difference in their absolute values.

Finally, we will compute the percentage of variance captured by each of the first six SVD modes and 
plot the first six SVD modes. This will give us an idea of how much information is retained by using 
these modes to represent the images instead of the original matrix $X$.

## Theoretical Background:
To successfully complete this assignment, a solid foundation in linear algebra and its applications 
is necessary. Specifically, knowledge of matrix operations, eigenvalues and eigenvectors, and 
singular value decomposition (SVD) is essential.

Matrix operations such as addition, subtraction, and multiplication are fundamental to this 
assignment. We will use dot products and element-wise multiplication of matrices to compute the 
correlation matrix and to find the eigenvectors and singular values of the matrix $X$. Additionally, 
knowledge of matrix transposition and reshaping will be useful in preparing the data for analysis and 
visualization.

Eigenvalues and eigenvectors play a critical role in analyzing the correlation matrix and finding the 
first six eigenvectors with the largest magnitude eigenvalue. Eigenvectors represent directions along 
which a transformation acts only by stretching or shrinking, and eigenvalues represent the magnitude 
of the stretch or shrink in that direction. In this assignment, we will use the eigenvectors of the 
correlation matrix to find the principal components of the data, which can be used to reduce the 
dimensionality of the dataset while retaining most of the information.

Singular value decomposition (SVD) is another key concept for this assignment. SVD decomposes a 
matrix into three components: $U$, $Σ$, and $V*$, where $U$ and $V*$ are orthogonal matrices and $Σ$ 
is a diagonal matrix with the singular values of the original matrix on its diagonal. SVD is useful 
for finding the principal components of a dataset and for compressing the data by retaining only a 
subset of the singular values.

## Algorithm Implementation and Development:
This homework assignment works around a dataset imported through the following lines of code:
```
import numpy as np
from scipy.io import loadmat
results = loadmat(’yalefaces.mat’)
X = results[’X’]
```

Completion of this project and subsequent development and implementation of the algorithm was 
accomplished through Python as our primary programming language. 

### Code Description
The code is written in Python and uses the following libraries:  
* `numpy` for numerical computing  
* `matplotlib` for data visualization  
* `math` for mathematical functions  
* `random` for random generation
* `scipy` for curve fitting

#### Problem (a): Computing Correlation Matrix using Dot Product
In this problem, a 100x100 correlation matrix $C$ is computed by computing the dot product 
(correlation) between the first 100 images in the matrix $X$. $X$ is an array consisting of 165 
grayscale images of human faces from the Yale Face Database. These images have been preprocessed and 
flattened to 1024-dimensional feature vectors.

First, the `scipy` library is used to load the yalefaces.mat file which contains the matrix $X$. The 
first 100 images of $X$ are selected by slicing the matrix. The numpy function `np.matmul()` is used 
to compute the dot product between the image vectors. This returns the correlation between each pair 
of images in a 100x100 symmetric matrix $C$. The plot of the correlation matrix is shown using the 
pcolor function of the matplotlib library. The color of each element of the plot represents the 
strength of correlation between the corresponding images.

```
images = X[:, :100]
C = np.matmul(images.T, images)
```
#### Problem (b): Identifying Highly Correlated and Uncorrelated Images
In this problem, the most highly correlated and most uncorrelated pairs of images are identified from 
the correlation matrix computed in Problem (a), and then the corresponding faces are plotted.

The indices of the most highly correlated pair and the least correlated pair are found by finding the 
locations of the maximum and minimum values in the correlation matrix $C$ using `np.argwhere()`. The 
pair with the highest correlation has the second highest value (since the highest value corresponds 
to self-correlation), and the pair with the least correlation has the second smallest value.

For each pair of images, the corresponding face images are plotted side by side. The faces are 
plotted using the `imshow()` function from the `matplotlib` library.

```
most_correlated = np.argwhere(C == np.sort(C.flatten())[-3])
least_correlated = np.argwhere(C == np.sort(C.flatten())[1])

print (most_correlated[0], least_correlated[0])
```
#### Problem (c): Computing Correlation Matrix for Subset of Images
This problem is similar to Problem (a), except the correlation matrix is now computed between a 
different set of 10 images.

A list of 10 image indices is specified and the corresponding images are extracted from the $X$ 
matrix. The images are then used to compute the correlation matrix $C$, which is again plotted using 
the `pcolor()` function.

```
images = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
image_list = X[:, np.subtract(images, 1)]

C = np.ndarray((10, 10))
C = np.matmul(image_list.T, image_list)
```
#### Problem (d): Finding the First Six Eigenvectors of $Y = XX^T$
This problem involves finding the first six eigenvectors with the largest magnitude eigenvalues for 
the matrix $Y = XX^T$, where $X$ is the same matrix used in previous problems.

The dot product of $X$ with its transpose is computed to obtain the $Y$ matrix. The numpy function 
`np.linalg.eigh()` is used to compute the eigenvalues and eigenvectors of $Y$. The eigenvalues and 
eigenvectors are sorted in descending order of eigenvalue magnitude using `np.argsort()[::-1]`.

The six eigenvectors with the largest magnitude eigenvalues are extracted and stored in the $W$ 
matrix. The first eigenvector is stored in the `v_1` vector, which is printed.

```
Y = np.dot(X, X.T)
eigenvalues, eigenvectors = np.linalg.eigh(Y)

W = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, W]

v_1 = eigenvectors[:, 0]
```
#### Problem (e): Finding the First Six Principal Component Directions using SVD
This problem involves computing the first six principal component directions of the matrix $X$ using 
Singular Value Decomposition (SVD).

The numpy function `np.linalg.svd()` is used to perform the SVD of $X$. The output of the function 
consists of three matrices: $U$, $S$, and $V$, where $U$ and $V$ are unitary matrices, and $S$ is a 
diagonal matrix of singular values. The first six principal component directions are given by the 
first six rows of the $V$ matrix.

The first six principal component directions are printed.

```
U, S, V = np.linalg.svd(X, full_matrices = False)
first_six = V[:6, :]
```
#### Problem (f): Comparing First Eigenvector and First SVD Mode
In problem (d), we calculated the first eigenvector `v_1`, using the covariance matrix of the image 
data matrix $X$. In problem (e), we obtained the first SVD mode, `u_1`, by decomposing the data 
matrix $X$ into its singular values and singular vectors. 

In this problem, we compare the first eigenvector `v_1` and the first SVD mode `u_1` and compute the 
norm of the difference between their absolute values. The code first extracts the first value of the 
SVD mode $U$ `u_1`, by using the slice operation on the matrix $U$. Then, it computes the norm of the 
difference between the absolute values of the first eigenvector `v_1` and the first SVD mode `u_1` 
using the `np.linalg.norm` function. The result is printed as the norm of the difference of absolute 
values of `v_1` and `u_1`.

```
u_1 = U[:, 0]
norm_of_difference = np.linalg.norm(np.abs(v_1) - np.abs(u_1))
```
#### Problem (g): Computing Variance Captured by Each of the First 6 SVD Modes and Plotting Them
In this problem, we compute the percentage of variance captured by each of the first six SVD modes 
and plot the first six SVD modes. The code first computes the percentage of variance captured by each 
of the first six SVD modes using the formula `((S[:6] ** 2) / np.sum(S ** 2)) * 100`. It then prints 
the percentage of variance captured by each SVD mode from 1 to 6 using a for loop. Finally, the code 
plots the first six SVD modes using a for loop and the `imshow()` function in `matplotlib`. 

The modes are reshaped to the original image shape and transposed to display the image in the correct 
orientation. The plot is displayed in a 2x3 grid with each plot displaying the image of the 
corresponding SVD mode with a title "SVD Mode k", where k is the mode number from 0 to 5. The plot is 
displayed using the `plt.show()` function.

## Computational Results:

### Usage
To run the code, simply run the Python file `hw2.py` in any Python environment. The output will be 
printed to the console and displayed in a pop-up window. The `matplotlib` library is required to 
display the images in the form of a plot. 

#### Problem (a): Computing Correlation Matrix using Dot Product
The resultant dot product (correlation) matrix between the first 100 images are plotted as followed:  
![download](https://user-images.githubusercontent.com/125385468/232674818-baf7ce66-d67c-465b-96e6-94afe61e22dc.png)

#### Problem (b): Identifying Highly Correlated and Uncorrelated Images
Using the first part of the program to determine the pairs of highly correlated and uncorrelated images: 
```
most_correlated = np.argwhere(C == np.sort(C.flatten())[-3])
least_correlated = np.argwhere(C == np.sort(C.flatten())[1])

print (most_correlated[0], least_correlated[0])
```
We could determine that the pairs `[86 88]` and `[54 64]` represent the indices of the images we are looking for. 

Plotting the following images yield:  
![download](https://user-images.githubusercontent.com/125385468/232675169-a2d1cddd-13b8-47db-b0f5-4176f676c089.png)

and  
![download](https://user-images.githubusercontent.com/125385468/232675206-9cf1c489-db47-4a41-a928-71bfea5d6583.png)

#### Problem (c): Computing Correlation Matrix for Subset of Images
The resultant dot product (correlation) matrix between the first 10 images are plotted as followed:  
![download](https://user-images.githubusercontent.com/125385468/232675344-552f1047-e830-4f7a-b9a7-c3a4ddbce12c.png)

#### Problem (d): Finding the First Six Eigenvectors of $Y=XX^T$
The resultant 6 eigenvectors with the largest magnitude eigenvalue are determined to be:
```
[[-0.02384327  0.04535378 -0.05653196 ... -0.00238077  0.0015886
  -0.00041024]
 [-0.02576146  0.04567536 -0.04709124 ...  0.00265168 -0.00886967
   0.0047811 ]
 [-0.02728448  0.04474528 -0.0362807  ... -0.00073077  0.00706009
  -0.00678472]
 ...
 [-0.02082937 -0.03737158 -0.06455006 ... -0.0047683  -0.00596037
  -0.0032901 ]
 [-0.0193902  -0.03557383 -0.06196898 ... -0.00173228 -0.00175508
  -0.00131795]
 [-0.0166019  -0.02965746 -0.05241684 ...  0.00458062  0.00266653
   0.00168849]]
```

#### Problem (e): Finding the First Six Principal Component Directions using SVD
The first six principal component directions are determined to be:
```
[[-0.01219331 -0.00215188 -0.01056679 ... -0.02177117 -0.03015309
  -0.0257889 ]
 [-0.01938848 -0.00195186  0.02471869 ...  0.04027773  0.00219562
   0.01553129]
 [ 0.01691206  0.00143586  0.0384465  ...  0.01340245 -0.01883373
   0.00643709]
 [ 0.0204079  -0.01201431  0.00397553 ... -0.01641295 -0.04011563
   0.02679029]
 [-0.01902342  0.00418948  0.0384026  ... -0.01092512  0.00087341
   0.01260435]
 [-0.0090084  -0.00624237  0.01580824 ... -0.00977639  0.00090316
   0.00304479]]
```

#### Problem (f): Comparing First Eigenvector and First SVD Mode
The norm of the difference of absolute values between `v_1` and `u_1` was determined to be:
```
7.394705201660225e-16
```

#### Problem (g): Computing Variance Captured by Each of the First 6 SVD Modes and Plotting Them
Computing the percentage of variance captured by each of the first 6 SVD modes yielded:
```
Percentage of variance captured by each SVD mode 1: 72.93%
Percentage of variance captured by each SVD mode 2: 15.28%
Percentage of variance captured by each SVD mode 3: 2.57%
Percentage of variance captured by each SVD mode 4: 1.88%
Percentage of variance captured by each SVD mode 5: 0.64%
Percentage of variance captured by each SVD mode 6: 0.59%
```

In addition, plotted SVD nodes were as followed:  
![download](https://user-images.githubusercontent.com/125385468/232675682-a428b9fe-b24c-483a-9631-1cf849293455.png)

## Summary and Conclusions:
In this assignment, we learned how to perform data analysis on a set of images using linear algebra 
concepts and Python programming. Specifically, we computed the correlation matrix between images, 
found the most highly correlated and uncorrelated images, computed the eigenvectors and singular 
values of the matrix, and analyzed the percentage of variance captured by the first six SVD modes.

We applied our knowledge of matrix operations, eigenvalues and eigenvectors, and singular value 
decomposition to perform these analyses. We used dot products, element-wise multiplication, and 
reshaping of matrices to prepare the data for analysis and visualization. We computed the eigenvalues 
and eigenvectors of the correlation matrix to find the principal components of the data. We also 
performed SVD to compress the data by retaining only a subset of the singular values.

In conclusion, this assignment provided a valuable opportunity to apply theoretical concepts to real-
world data and gain practical experience in data analysis. The knowledge and skills gained from this 
assignment can be applied to a wide range of fields and applications, such as image processing, 
machine learning, and data compression. By continuing to explore and expand upon these concepts and 
techniques, we can continue to develop our understanding and expertise in linear algebra and data 
analysis. 
