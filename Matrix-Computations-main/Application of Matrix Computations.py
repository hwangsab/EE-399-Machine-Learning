#!/usr/bin/env python
# coding: utf-8

# # EE 399 Introduction to Machine Learning: HW 2 Submission
# ### Sabrina Hwang

# In[41]:


# GitHub HW2: https://github.com/hwangsab/EE-399-HW2


# In[42]:


import numpy as np
import matplotlib.pyplot as plt
import math as math
import random
import scipy

from scipy.io import loadmat
from scipy.sparse.linalg import eigs
from numpy import linalg


# In[43]:


X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])


# ### Problem (a)
# Compute a 100 × 100 correlation matrix C where you will compute the dot product (correlation) between the first 100 images in the matrix X.

# In[44]:


results = loadmat('yalefaces.mat')
X = results ['X']


# In[65]:


# Take the first 100 images from the data matrix
images = X[:, :100]

# Compute the correlation matrix
C = np.matmul(images.T, images)

# Plot the correlation matrix using pcolor
plt.figure()
plt.title("Dot Product (Correlation) Between First 100 Images")
plt.xlabel("Image 1")
plt.ylabel("Image 2")

plt.pcolor(C, cmap='jet')
color_bar = plt.colorbar()
color_bar.set_label('Correlation Coefficient')

plt.show()


# ### Problem (b)
# From the correlation matrix for part (a), which two images are most highly correlated? Which are most uncorrelated? Plot these faces

# In[74]:


most_correlated = np.argwhere(C == np.sort(C.flatten())[-3])
least_correlated = np.argwhere(C == np.sort(C.flatten())[1])

print (most_correlated[0], least_correlated[0])

# Plot image pairs for highest correlation
fig = plt.figure(figsize=(10, 5))
fig.suptitle('Images with Highest Correlation')

ax1 = plt.subplot(121)
ax1.set_title('Image 86'.format(most_correlated[0]))
ax1.imshow(X[:, 86].reshape((32, 32), order = 'F').T)

ax2 = plt.subplot(122)
ax2.set_title('Image 88'.format(most_correlated[1]))
ax2.imshow(X[:, 88].reshape((32, 32), order = 'F').T)

# Plot image pairs for lowest correlation
fig = plt.figure(figsize=(10, 5))
fig.suptitle('Images with Lowest Correlation')

ax1 = plt.subplot(121)
ax1.set_title('Image 54'.format(least_correlated[0]))
ax1.imshow(X[:, 54].reshape((32, 32), order = 'F').T)

ax2 = plt.subplot(122)
ax2.set_title('Image 64'.format(least_correlated[1]))
ax2.imshow(X[:, 64].reshape((32, 32), order = 'F').T)


# ### Problem (c)
# Repeat part (a) but now compute the 10 × 10 correlation matrix between images and plot the correlation matrix between them.

# In[47]:


images = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
image_list = X[:, np.subtract(images, 1)]

C = np.ndarray((10, 10))

# Compute the correlation matrix
C = np.matmul(image_list.T, image_list)

# Plot the correlation matrix using pcolor
plt.figure()
plt.pcolor(C, cmap='jet')
plt.title("Dot Product (Correlation) Between First 100 Images")
plt.xlabel("Image 1")
plt.ylabel("Image 2")
plt.colorbar()
plt.show()


# ### Problem (d)
# Create the matrix Y = XX^T and find the first six eigenvectors with the largest magnitude eigenvalue.

# In[80]:


Y = np.dot(X, X.T)

eigenvalues, eigenvectors = np.linalg.eigh(Y)

W = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, W]

v_1 = eigenvectors[:, 0]

print(eigenvectors)


# ### Problem (e)
# SVD the matrix X and find the first six principal component directions.

# In[79]:


U, S, V = np.linalg.svd(X, full_matrices = False)

# First six principal component directions
first_six = V[:6, :]

print("First six principal component directions:")
print(first_six)


# ### Problem (f)
# Compare the first eigenvector v1 from (d) with the first SVD mode u1 from (e) and compute the norm of difference of their absolute values.

# In[82]:


# First value of SVD mode U
u_1 = U[:, 0]

# Computing norm of difference
norm_of_difference = np.linalg.norm(np.abs(v_1) - np.abs(u_1))

print("Norm of the difference of absolute values of v_1 and u_1:")
print (norm_of_difference)


# ### Problem (g)
# Compute the percentage of variance captured by each of the first 6 SVD modes. Plot the first 6 SVD modes

# In[84]:


# Compute and print percentage of variance
variance_ratios = (S[:6] ** 2) / np.sum(S ** 2) * 100

for i, variance_ratio in enumerate(variance_ratios, 1):
    print(f"Percentage of variance captured by each SVD mode {i}: {variance_ratio:.2f}%")

# Plot the first 6 SVD modes
fig = plt.figure(figsize=(15, 10))
for k in range(6):
        mode = dir[:, k].reshape((32, 32), order='F').T
        axes = fig.add_subplot(2, 3, k + 1)
        axes.imshow(mode)
        axes.set_title(f"SVD Mode {k}")


# In[ ]:




