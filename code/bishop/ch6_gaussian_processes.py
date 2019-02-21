#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 09:10:09 2019

@author: tommy
"""

import numpy as np
import matplotlib.pyplot as plt


def kernel_function(arg, gaussian_weight=1, exponential_weight=0):
    """
    Apply a kernel function.
    """
    # Create the kernel matrix K
    K = gaussian_weight * np.exp(-50 * (arg) ** 2)
    K += exponential_weight * np.exp(-0.1 * np.abs(arg))
    return K


def kernel_matrix(arr1, arr2, beta_inv, gaussian_weight, exponential_weight):
    """
    Vectorized kernel computation.
    Computes differences in a matrix, and then applies a vectorized function.
    
    Shape of K is len(arr1), len(arr2)
    """

    # Create outer product, calculation for kernels
    arg1 = np.outer(arr1, np.ones_like(arr2))
    arg2 = np.outer(np.ones_like(arr1), arr2)

    # Use a kernel consisting of three parts: gaussian, exponential, linear
    K = kernel_function(
        arg1 - arg2,
        gaussian_weight=gaussian_weight,
        exponential_weight=exponential_weight,
    )

    if len(arr1) == len(arr2):
        K += np.diag(np.ones_like(arr1)) * beta_inv

    assert K.shape == (len(arr1), len(arr2))

    return K


# ---------------------------------------
# ----- SAMPLE GAUSSIAN PROCESS ---------
# ---------------------------------------
beta_inv = 1e-6
gaussian_weight = 1
exponential_weight = 0

# Point to sample on
x = np.linspace(0, 1, num=2 ** 9)

K = kernel_matrix(x, x, beta_inv, gaussian_weight, exponential_weight)
plt.title("Kernel matrix")
plt.imshow(K)
plt.show()

# Check that eigenvalues are positive
eigs, *_ = np.linalg.eig(K)
assert np.all(eigs >= 0)

# Sample from multivariate normal
mean = np.zeros_like(x)
samples = np.random.multivariate_normal(mean, K)
plt.title("Sampled Gaussian process")
plt.plot(x, samples)
plt.grid(True)
plt.show()


# ---------------------------------------
# ----- GAUSSIAN PROCESS REGRESSION -----
# ---------------------------------------
# np.random.seed(123)

x_data = np.random.rand(2 ** 4)
x_data = np.sort(x_data)
t = np.sin(x_data * 10) + np.random.randn(len(x_data)) / 5

x_smooth = np.linspace(-0.5, 1.2, num=2 ** 10)
y_smooth = np.sin(x_smooth * 10)

plt.plot(x_smooth, y_smooth, color="blue", label="True function")
plt.scatter(x_data, t, color="blue", label="Samples with noise", alpha=0.5, s=25)

beta_inv = 1e-2
gaussian_weight = 1
exponential_weight = 1

# --------------------------------------------------------
# ----- PREDICT THE MEANS: Equation (6.66) in Bishop -----
# --------------------------------------------------------

# This is a matrix version of k^T in Equation (6.66), to predict on many points
K_T = kernel_matrix(x_smooth, x_data, beta_inv, gaussian_weight, exponential_weight)

# Instead of computing mean = K_T C^{-1} t directly with matrix inversion,
# we first solve t = C * x for x, and then set m = K_new * x
C = kernel_matrix(x_data, x_data, beta_inv, gaussian_weight, exponential_weight)
x = np.linalg.solve(C, t)
mean = K_T @ x

# -----------------------------------------------------------
# ----- PREDICT THE VARIANCE: Equation (6.67) in Bishop -----
# -----------------------------------------------------------

c = kernel_function(x_smooth - x_smooth, gaussian_weight, exponential_weight)

# Predict the variances. This is Equation (6.67) on page 308 in Bishop
# Solve C * X = K. The result is X = (C^{-1}k1, C^{-1}k2, , C^{-1}k3, , ...),
# a matrix of size N times M, where N is old data points, M is new grid data
X = np.linalg.solve(C, K_T.T)
predicted_variance = np.empty_like(mean)

# Look over each point and apply quadratic form
for i in range(len(predicted_variance)):
    predicted_variance[i] = c[i] - K_T[i, :] @ X[:, i] + beta_inv


plt.plot(x_smooth, mean, label="Predicted means", color="red")
plt.fill_between(
    x_smooth,
    y1=mean - predicted_variance,
    y2=mean + predicted_variance,
    label="Predicted variance",
    color="red",
    alpha=0.2,
)

plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig('gaussian_process_regression.pdf')
plt.show()
