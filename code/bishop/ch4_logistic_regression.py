#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script naively implements the algorithm sketched in section 4.3.3
in Bishop, titled 'Iterative reweighted least squares'.
"""

import numpy as np
import time
import matplotlib.pyplot as plt


def generate_data(N, random_seed=123):
    np.random.seed(random_seed)
    data1 = np.hstack(
        (np.random.randn(N, 1) + 1, np.random.randn(N, 1) + 1, np.ones((N, 1)))
    )
    data2 = np.hstack(
        (np.random.randn(N, 1), np.random.randn(N, 1), np.ones((N, 1)) * 0)
    )

    return np.vstack((data1, data2))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Generate data points
# If the data is too separated, the algorithm struggles with convergence
N = 1000
data = generate_data(N)

# Prepare for the algorithm, add a bias column of 1s
t = data[:, -1].copy()
phi = data[:, :-1].copy()
phi = np.hstack((phi, np.ones((N * 2, 1))))

# Initial weight
w = np.array([1, 1, 1])

for iter_count in range(1, 100 + 1):

    # This computes the y_ns needed in Equation (4.98)
    y_n = sigmoid(phi @ w)

    # This is Equation (4.98) in Bishop for R_nn
    R_dia_elems = y_n * (1 - y_n)
    R = np.diag(R_dia_elems)
    R_inv = np.diag(1 / R_dia_elems)

    # This is the weight update rule, given by Equation (4.99)
    z = phi @ w - R_inv @ (y_n - t)

    # Notice the order of multiplication here
    phi_T_R_z = phi.T @ (R @ z)

    # Compute the new weight, without explicitly computing inverse
    w_new = np.linalg.solve(phi.T @ R @ phi, phi_T_R_z)

    # Convergence criterion
    print("Difference:", np.linalg.norm(w_new - w))
    if np.linalg.norm(w_new - w) < 1e-5:
        break
    w = w_new.copy()

    # Create a plot
    plt.title(f"Iteration {iter_count}. w = {w}")
    class1 = data[data[:, 2] == 1]
    plt.scatter(class1[:, 0], class1[:, 1], alpha=0.2)
    class2 = data[data[:, 2] == 0]
    plt.scatter(class2[:, 0], class2[:, 1], alpha=0.2)
    plt.arrow(0, 0, w[0], w[1])
    plt.show()
    print()
    time.sleep(0.5)
