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


def stochastic_simulated_annealing(W, temp_init=10, temp_decay=0.9, 
                                   main_iterations=20, sub_iterations=None):
    """
    Stochastic simulated annealing to minimize E = -0.5 s.T @ W @ s
    """
    
    # The matrix W should be symmetric
    assert np.all(W.T == W)
    
    # The number of nodes
    num_nodes = W.shape[0]
    
    # If sub-iterations are not set, sample 5 times number of nodes
    sub_iterations = sub_iterations or num_nodes * 5

    # Generate a random starting position, a vector with 1 or -1
    s = np.random.randint(0, 2, num_nodes) * 2 - 1

    for main_iteration in range(main_iterations):
    
        
        for sub_iteration in range(sub_iterations):
            
            random_node = np.random.randint(num_nodes)
            
    
            new_s = s.copy()
            new_s[random_node] = -new_s[random_node]
    
    
            #print('---------------------------')
            #print(random_node)
            #print(W[random_node, :])
            #print(s)
            old_energy = s[random_node] * ((W[random_node, :] * s).sum())
            new_energy = -old_energy
            
            #print(s, new_s.T @ W @ new_s,  s.T @ W @ s)
            if new_energy < old_energy:
                s = new_s.copy()
                yield s, temp_init
            else:
                #print(np.exp((old_energy - new_energy) / temperature))
                rand_num = np.random.rand()
                energy_diff = old_energy - new_energy
                #print(energy_diff)
                if np.exp(energy_diff/ temp_init) > rand_num:
                    s = new_s.copy()
                    yield s, temp_init
                else:
                    yield s, temp_init

            
        temp_init *= temp_decay
    
    
# ----------------------------------------------
# ----------- PROBLEM SETUP --------------------
# ----------------------------------------------
        
W = np.array([[0, 5, -3, 4, 4, 1],
              [0, 0, -1, 2, -3, 1],
              [0, 0, 0, 2, 2, 0],
              [0, 0, 0, 0, 3, -3],
              [0, 0, 0, 0, 0, 5],
              [0, 0, 0, 0, 0, 0]])

W = W + W.T

np.random.seed(42)


# ----------------------------------------------
# ----------- RUN SIMULATED ANNEALING ----------
# ----------------------------------------------   
logged_temperatures, logged_energy = [], []

for s, temp in stochastic_simulated_annealing(W, temp_init=10, temp_decay=0.9): 

    logged_temperatures.append(temp)
    logged_energy.append(s.T @ W @ s)


# ----------------------------------------------
# ----------- PLOT THE RESULTS -----------------
# ----------------------------------------------   
    
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
ax = plt.subplot()    
line1, = ax.plot(logged_temperatures, color=COLORS[0])
ax.set_ylabel('Temperature')

ax.grid(zorder=-15)

ax2 = ax.twinx()
line2, = ax2.plot(logged_energy, color=COLORS[1])
ax2.set_ylabel('Energy')
ax.legend((line1, line2), ('Temperature', 'Energy'))
plt.show()
        
        
    