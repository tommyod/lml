#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script naively implements Mixtures of Bernoulli distributions, also called
"latent class analysis". The implementation follows Section 9.3.3 in the book
"Pattern Recognition and Machine Learning" by Bishop.

There are D binary variables x_i, comprising a vector x of dimension D.
Each x_is is governed by a Bernoulli distribution, so that

p(x | u) = \prod_i u_i^{x_i} * (1 - u_i)^{1 - x_i}.         (Eqn 9.44 in Bishop)

In other words, x_i and x_j are independent, and the probability that x_i = 1 is u_i.
The mixture model is given by

p(x) = \sum_k \pi_k p(x | u_k)                              (Eqn 9.47 in Bishop)

The log likelihood is maximized with respect to \pi_k and u_k.





%timeit generate_data(N=50000, k=10, dim=10)  165 ms ± 1.68 ms
"""

import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator


def generate_data(N, k, dim=6, seed=123):
    """
    Ancestral sampling to generate data.
    """
    
    np.random.seed(seed)
    
    # Generate mixing coefficients. They are in the range [0, 1] and sum to 1.
    pi_ks = np.random.rand(k)
    pi_ks = pi_ks / np.sum(pi_ks)
    assert np.allclose(np.sum(pi_ks), 1)
    
    # Generate mean vectors u_k, store these in a list. 
    # Each u_k has entries in the range [0, 1].
    u_ks = [np.random.rand(dim) for j in range(k)]
    assert all(np.all((u_k <= 1) & (u_k >= 0)) for u_k in u_ks)
    
    # print('The mean vectors for the mixtures are:', u_ks)
    
    data = np.empty((N, dim))
    
    mix_coeff = np.random.choice(a=list(range(k)), size=N, 
                                 replace=True, p=pi_ks)
    
    for n, chosen_k in enumerate(mix_coeff):
        
        draw = np.random.rand(dim)
        draw = np.where(draw > u_ks[chosen_k], 1, 0)
        data[n, :] = draw
    
    # print(mix_coeff)
    
    np.random.rand(4)
    
    return data
    
    
    
    
    
data = generate_data(N=50, k=3, dim=10)

print(data)
    
    


