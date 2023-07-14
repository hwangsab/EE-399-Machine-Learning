#!/usr/bin/env python
# coding: utf-8

# # EE 399 Introduction to Machine Learning: HW 1 Submission
# ### Sabrina Hwang

# In[138]:


# GitHub HW1: https://github.com/hwangsab/EE-399-HW1


# In[129]:


import numpy as np
import matplotlib.pyplot as plt
import math as math
import warnings

from scipy.optimize import curve_fit
warnings.filterwarnings('ignore', 'Polyfit may be poorly conditioned')


# In[130]:


X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])


# ### II (i)
# Code to find the minimum error and determine the parameters A, B, C, D of the given dataset

# In[131]:


def func(x, A, B, C, D):
    return A*np.cos(B*x) + C*x + D

popt, pcov = curve_fit(func, X, Y)

A, B, C, D = popt

error = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

plt.scatter(X, Y, label='Data points')
plt.plot(X, func(X, *popt), 'r-', label='Fitted curve')
plt.title("Least-Square Fit")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.legend()
plt.show()

print("Minimum error:", error)
print("Optimized values:")
print("     A =", A)
print("     B =", B)
print("     C =", C)
print("     D =", D)


# ### II (ii) 
# With the results of (i), fix two of the parameters and sweep through values of the other two parameters to generate a 2D loss (error) landscape. Do all combinations of two fixed parameters and two swept parameters. You can use something like pcolor to visualize the results in a grid. How many minima can you find as you sweep through parameters?

# In[137]:


fig, axs = plt.subplots(3, 2, figsize=(10, 12))
fig.suptitle('2D Error Landscape')
plt.subplots_adjust(hspace=0.5)

A = 2.1717269828948855
B = 0.909325796914226

# Fix parameters A and B, sweep C and D
C_range = np.linspace(-5, 5, 100)
D_range = np.linspace(30, 60, 100)
C_grid, D_grid = np.meshgrid(C_range, D_range)
error_grid = np.zeros_like(C_grid)
for i in range(len(C_range)):
    for j in range(len(D_range)):
        C = C_range[i]
        D = D_range[j]
        error_grid[j,i] = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

plt.subplot(321)
plt.pcolor(D_grid, C_grid, error_grid)
plt.xlabel('D')
plt.ylabel('C')
plt.title('Fixed parameters A and B (A = {:.1f}, B = {:.1f})'.format(A, B))
plt.colorbar()

min_pos = np.argmin(error_grid)
min_C, min_D = np.unravel_index(min_pos, error_grid.shape)
min_error = error_grid[min_C, min_D]
print(f"Fixed parameters A and B Minimum error: {min_error:.2f} at C = {C_range[min_C]:.2f}, D = {D_range[min_D]:.2f}")

# Fix parameters A and C, sweep B and D
B_range = np.linspace(0.01, 1.0, 100)
D_range = np.linspace(30, 60, 100)
B_grid, D_grid = np.meshgrid(B_range, D_range)
error_grid = np.zeros_like(B_grid)
for i in range(len(B_range)):
    for j in range(len(D_range)):
        B = B_range[i]
        D = D_range[j]
        error_grid[j,i] = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

plt.subplot(322)
plt.pcolor(D_grid, B_grid, error_grid)
plt.xlabel('D')
plt.ylabel('B')
plt.title('Fixed parameters A and C (A = {:.1f}, C = {:.1f})'.format(A, C))
plt.colorbar()

min_pos = np.argmin(error_grid)
min_B, min_D = np.unravel_index(min_pos, error_grid.shape)
min_error = error_grid[min_B, min_D]
print(f"Fixed parameters A and C Minimum error: {min_error:.2f} at B = {B_range[min_B]:.2f}, D = {D_range[min_D]:.2f}")

# Fix parameters A and D, sweep B and C
B_range = np.linspace(0.01, 1.0, 100)
C_range = np.linspace(-5, 5, 100)
B_grid, C_grid = np.meshgrid(B_range, C_range)
error_grid = np.zeros_like(B_grid)
for i in range(len(B_range)):
    for j in range(len(C_range)):
        B = B_range[i]
        C = C_range[j]
        error_grid[j,i] = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

plt.subplot(323)
plt.pcolor(C_grid, B_grid, error_grid)
plt.xlabel('C')
plt.ylabel('B')
plt.title('Fixed parameters A and D (A = {:.1f}, D = {:.1f})'.format(A, D))
plt.colorbar()

min_pos = np.argmin(error_grid)
min_B, min_C = np.unravel_index(min_pos, error_grid.shape)
min_error = error_grid[min_B, min_C]
print(f"Fixed parameters A and D Minimum error: {min_error:.2f} at B = {B_range[min_B]:.2f}, C = {C_range[min_C]:.2f}")

# Fix parameters B and C, sweep A and D
A_range = np.linspace(0.1, 2.0, 100)
D_range = np.linspace(30, 60, 100)
A_grid, D_grid = np.meshgrid(A_range, D_range)
error_grid = np.zeros_like(A_grid)
for i in range(len(A_range)):
    for j in range(len(D_range)):
        A = A_range[i]
        D = D_range[j]
        error_grid[j,i] = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

plt.subplot(324)
plt.pcolor(D_grid, A_grid, error_grid)
plt.xlabel('D')
plt.ylabel('A')
plt.title('Fixed parameters B and C (B = {:.1f}, C = {:.1f})'.format(B, C))
plt.colorbar()

min_pos = np.argmin(error_grid)
min_A, min_D = np.unravel_index(min_pos, error_grid.shape)
min_error = error_grid[min_A, min_D]
print(f"Fixed parameters B and C Minimum error: {min_error:.2f} at A = {A_range[min_A]:.2f}, D = {D_range[min_D]:.2f}")

# Fix parameters B and D, sweep A and C
A_range = np.linspace(0.1, 2.0, 100)
C_range = np.linspace(-5, 5, 100)
A_grid, C_grid = np.meshgrid(A_range, C_range)
error_grid = np.zeros_like(A_grid)
for i in range(len(A_range)):
    for j in range(len(C_range)):
        A = A_range[i]
        C = C_range[j]
        error_grid[j,i] = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

plt.subplot(325)
plt.pcolor(C_grid, A_grid, error_grid)
plt.xlabel('C')
plt.ylabel('A')
plt.title('Fixed parameters B and D (B = {:.1f}, D = {:.1f})'.format(B, D))
plt.colorbar()

min_pos = np.argmin(error_grid)
min_A, min_C = np.unravel_index(min_pos, error_grid.shape)
min_error = error_grid[min_A, min_C]
print(f"Fixed parameters B and D Minimum error: {min_error:.2f} at A = {A_range[min_A]:.2f}, C = {C_range[min_C]:.2f}")

# Fix parameters C and D, sweep A and B
A_range = np.linspace(0.1, 2.0, 100)
B_range = np.linspace(0.01, 1.0, 100)
A_grid, B_grid = np.meshgrid(A_range, B_range)
error_grid = np.zeros_like(A_grid)
for i in range(len(A_range)):
    for j in range(len(B_range)):
        A = A_range[i]
        B = B_range[j]
        error_grid[j,i] = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))

plt.subplot(326)
plt.pcolor(B_grid, A_grid, error_grid)
plt.xlabel('B')
plt.ylabel('A')
plt.title('Fixed parameters C and D (C = {:.1f}, D = {:.1f})'.format(C, D))
plt.colorbar()

min_pos = np.argmin(error_grid)
min_A, min_B = np.unravel_index(min_pos, error_grid.shape)
min_error = error_grid[min_A, min_B]
print(f"Fixed parameters C and D Minimum error: {min_error:.2f} at A = {A_range[min_A]:.2f}, B = {B_range[min_B]:.2f}")

plt.show()


# ### II (iii) 
# Using the first 20 data points as training data, fit a line, parabola and 19th degree
# polynomial to the data. Compute the least-square error for each of these over the training
# points. Then compute the least square error of these models on the test data which are
# the remaining 10 data points.

# In[133]:


# data
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

# split the data into training and test sets
X_train = X[:20]
Y_train = Y[:20]
X_test = X[20:]
Y_test = Y[20:]

# fit line, parabola, and 19th degree polynomial
line_coeffs = np.polyfit(X_train, Y_train, 1)
parabola_coeffs = np.polyfit(X_train, Y_train, 2)
poly_coeffs = np.polyfit(X_train, Y_train, 19)

# compute predictions on train and test data
Y_line_train = np.polyval(line_coeffs, X_train)
Y_parabola_train = np.polyval(parabola_coeffs, X_train)
Y_poly_train = np.polyval(poly_coeffs, X_train)

Y_line_test = np.polyval(line_coeffs, X_test)
Y_parabola_test = np.polyval(parabola_coeffs, X_test)
Y_poly_test = np.polyval(poly_coeffs, X_test)

# compute least square error on train and test data
line_train_error = np.sum((Y_line_train - Y_train)**2)
parabola_train_error = np.sum((Y_parabola_train - Y_train)**2)
poly_train_error = np.sum((Y_poly_train - Y_train)**2)

line_test_error = np.sum((Y_line_test - Y_test)**2)
parabola_test_error = np.sum((Y_parabola_test - Y_test)**2)
poly_test_error = np.sum((Y_poly_test - Y_test)**2)

# plot data and fits
fig, ax = plt.subplots()
ax.plot(X_train, Y_train, 'bo', label='Training Data')
ax.plot(X_test, Y_test, 'ro', label='Test Data')
ax.plot(X_train, Y_line_train, 'g', label='Line Fit (Train)')
ax.plot(X_train, Y_parabola_train, 'c', label='Parabola Fit (Train)')
ax.plot(X_train, Y_poly_train, 'm', label='19th Degree Polynomial Fit (Train)')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()

# print least square errors
print("Line Train Error:", line_train_error)
print("Parabola Train Error:", parabola_train_error)
print("19th Degree Polynomial Train Error:", poly_train_error)
print()
print("Line Test Data Error:", line_test_error)
print("Parabola Test Data Error:", parabola_test_error)
print("19th Degree Polynomial Test Data Error:", poly_test_error)


# ### II (iv) 
# Repeat (iii) but use the first 10 and last 10 data points as training data. Then fit the
# model to the test data (which are the 10 held out middle data points). Compare these
# results to (iii)

# In[118]:


# data
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

# split the data into training and test sets
X_train = np.concatenate((X[:10], X[20:]))
Y_train = np.concatenate((Y[:10], Y[20:]))
X_test = X[10:20]
Y_test = Y[10:20]

# fit line, parabola, and 19th degree polynomial
line_coeffs = np.polyfit(X_train, Y_train, 1)
parabola_coeffs = np.polyfit(X_train, Y_train, 2)
poly_coeffs = np.polyfit(X_train, Y_train, 19)

# compute predictions on train and test data
Y_line_train = np.polyval(line_coeffs, X_train)
Y_parabola_train = np.polyval(parabola_coeffs, X_train)
Y_poly_train = np.polyval(poly_coeffs, X_train)

Y_line_test = np.polyval(line_coeffs, X_test)
Y_parabola_test = np.polyval(parabola_coeffs, X_test)
Y_poly_test = np.polyval(poly_coeffs, X_test)

# compute least square error on train and test data
line_train_error = np.sum((Y_line_train - Y_train)**2)
parabola_train_error = np.sum((Y_parabola_train - Y_train)**2)
poly_train_error = np.sum((Y_poly_train - Y_train)**2)

line_test_error = np.sum((Y_line_test - Y_test)**2)
parabola_test_error = np.sum((Y_parabola_test - Y_test)**2)
poly_test_error = np.sum((Y_poly_test - Y_test)**2)

# plot data and fits
fig, ax = plt.subplots()
ax.plot(X_train, Y_train, 'bo', label='Training Data')
ax.plot(X_test, Y_test, 'ro', label='Test Data')
ax.plot(X_train, Y_line_train, 'g', label='Line Fit (Train)')
ax.plot(X_train, Y_parabola_train, 'c', label='Parabola Fit (Train)')
ax.plot(X_train, Y_poly_train, 'm', label='19th Degree Polynomial Fit (Train)')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()

# print least square errors
print("Line Train Error:", line_train_error)
print("Parabola Train Error:", parabola_train_error)
print("19th Degree Polynomial Train Error:", poly_train_error)
print()
print("Line Test Data Error:", line_test_error)
print("Parabola Test Data Error:", parabola_test_error)
print("19th Degree Polynomial Test Data Error:", poly_test_error)


# Comparing questions II.iii and II.iv, the model generated that takes in the first 10 and last 10 data points as training data and the middle 10 data points as test data (from II.iv) has a lower minimized error. The magnitude at which these errors are different is that as the polynomial degree increases, the minimized error will decrease more. This is most likely because the model accounts for the shape of the beginning and the end, and because the dataset provided is relatively continuous, the model fits the data better if it is trained with points from the beginning and the end. 
