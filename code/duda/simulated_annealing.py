#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 08:47:26 2018

@author: tommy
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
import time


def stochastic_simulated_annealing(
    W, temp_init=10, temp_decay=0.9, main_iterations=10, sub_iterations=None
):
    """
    Stochastic simulated annealing to minimize E = s.T @ W @ s.
    See Algorithm 1 in Chapter 7 of Duda et al. for details.
    
    Parameters
    ----------
    W : Symmetric matrix of weights
    temp_init : initial temperature
    temp_decay : decay coefficient c such that T(k+1) = c * T(k)
    main_iterations : number of itertions where T is constant
    sub_iterations : number of iterations for each constant temperature.
                     defaults to  5 times the number of nodes
    
    """
    # The matrix W should be symmetric, and other tests
    assert np.all(W.T == W)
    assert temp_init > 0
    assert 0 <= temp_decay <= 1
    assert main_iterations > 0
    assert (sub_iterations is None) or (sub_iterations > 0)

    # The number of nodes
    num_nodes = W.shape[0]

    # If sub-iterations are not set, sample 5 times number of nodes
    sub_iterations = sub_iterations or num_nodes * 5

    # Generate a random starting position, a vector with 1 or -1
    s = np.random.randint(0, 2, num_nodes) * 2 - 1

    # The main loop - for each iteration in this loop, the temp is constant
    for main_iteration in range(main_iterations):

        # The sub - loop - for each iteration a new s_i is selected
        for sub_iteration in range(sub_iterations):

            # Select a random node
            random_node = np.random.randint(num_nodes)

            # If the annealing decides to proceed with this value, the
            # entry has it's sign flipped
            new_s = s.copy()
            new_s[random_node] = -new_s[random_node]

            # This computation is faster than s.T @ W @ s.
            # Energy diff = old - new = old - (-old) = 2 * old
            old_energy = s[random_node] * ((W[random_node, :] * s).sum())
            energy_diff = 2 * old_energy

            # If the energy goes down, keep the new configuration
            if energy_diff > 0:
                s = new_s.copy()
                yield s, temp_init

            # If the energy goes up, keep the new configuration sometimes
            else:
                if np.exp(energy_diff / temp_init) > np.random.rand():
                    s = new_s.copy()
                yield s, temp_init

        # End of outer loop - decrease the annealing temperature
        temp_init *= temp_decay


# ----------------------------------------------
# ----------- PROBLEM SETUP --------------------
# ----------------------------------------------

W = np.array(
    [
        [0, 5, -3, 4, 4, 1],
        [0, 0, -1, 2, -3, 1],
        [0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 3, -3],
        [0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0],
    ]
)

W = W + W.T

np.random.seed(42)


# ----------------------------------------------
# ----------- RUN SIMULATED ANNEALING ----------
# ----------------------------------------------
logged_temperatures, logged_energy = [], []

for s, temp in stochastic_simulated_annealing(
    W, temp_init=10, temp_decay=0.9, main_iterations=20
):

    logged_temperatures.append(temp)
    logged_energy.append(s.T @ W @ s)


# ----------------------------------------------
# ----------- PLOT THE RESULTS -----------------
# ----------------------------------------------

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.figure(figsize=(6, 3.5))
ax = plt.subplot()
line1, = ax.plot(logged_temperatures, color=COLORS[0])
ax.set_ylabel("Temperature")

ax.grid(zorder=-15)

ax2 = ax.twinx()
line2, = ax2.plot(logged_energy, color=COLORS[1])
ax2.set_ylabel("Energy")
ax.legend((line1, line2), ("Temperature", "Energy"))

plt.tight_layout()
# plt.savefig('../latex/figs/ch7_computer_ex2_a.pdf')

plt.show()


# ----------------------------------------------
# ----------- AVERAGE PROGRESS -----------------
# ----------------------------------------------

# Log energy decrease for every global (top level) simulation
globalsim_energy = []

for simulation in range(500):
    np.random.seed(simulation)

    logged_temperatures, logged_energy = [], []

    for s, temp in stochastic_simulated_annealing(
        W, temp_init=10, temp_decay=0.9, main_iterations=20
    ):

        logged_temperatures.append(temp)
        logged_energy.append(s.T @ W @ s)

    globalsim_energy.append(logged_energy)


# Create the plot
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.figure(figsize=(6, 3.5))

ax = plt.subplot()
ax.set_title("Average progress over 500 runs")
line1, = ax.plot(logged_temperatures, color=COLORS[0])
ax.set_ylabel("Temperature")

ax.grid(zorder=-15)

ax2 = ax.twinx()
line2, = ax2.plot(np.array(globalsim_energy).mean(axis=0), color=COLORS[1])
ax2.set_ylabel("Energy")
ax.legend((line1, line2), ("Temperature", "Energy"))

plt.tight_layout()
# plt.savefig('../latex/figs/ch7_computer_ex2_extra.pdf')

plt.show()
