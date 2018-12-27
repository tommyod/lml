#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 22:51:16 2018

@author: tommy
"""

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator


class BayesianBernoulli(BaseEstimator):
    """
    A bayesian Bernoulli distribution, where mu is modeled using the beta
    distribution (which is conjugate prior to the Bernoulli distribution).

    See Section 2.1 in Bishop.

    Examples
    --------
    # Initialize with a=1 and b=1, see Eqn (2.13)
    >>> model = BayesianBernoulli(1, 1)
    >>> model.mu.mean()  # With one virtual sample in each cat, the mean is 0.5
    0.5
    # Fit three new observations, updating the posterior for mu. Eqn (2.18)
    >>> model.fit([1, 1, 1])
    >>> model.mu.mean()
    0.8
    # The cumulative density function for mu can be evaluated.
    >>> list(model.mu.cdf([0, 0.5, 0.9]))
    [0.0, 0.0625, 0.6561]
    # Probability of 1, this is Eqn (2.19)
    >>> model.p.pmf(1)
    0.8
    """

    def __init__(self, virtual_ones=1, virtual_zeros=1):
        """
        Initilize a Bernoulli model.
        """
        self.virtual_ones = virtual_ones
        self.virtual_zeros = virtual_zeros

    def var(self):
        """
        The variance, computed using the expected value of the beta
        distribution over the mean.
        """
        p = self.mu.mean()
        return p * (1 - p)

    @property
    def p(self):
        """
        Initialize the Bernoulli distribution and return.
        """
        return stats.bernoulli(p=self.mu.mean())

    @property
    def mu(self):
        """
        Return the pdf over mu.
        """
        return stats.beta(a=self.virtual_ones, b=self.virtual_zeros)

    def fit(self, data):
        """
        Fit the model to data, i.e. update the posterior distribution.
        """
        data = np.asarray(data)
        ones = np.sum(data)
        zeros = len(data) - ones
        self.virtual_ones += ones
        self.virtual_zeros += zeros

    def evaluate(self):
        pass


if __name__ == "__main__":
    import pytest
    pytest.main(args=['.', '--doctest-modules', '-v', '--capture=sys'])

if __name__ == '__main__':

    # Import packages
    import matplotlib.pyplot as plt
    import random
    random.seed(42)

    # Create a model for coin tosses, see Section 2.1 in Bishop
    model = BayesianBernoulli(virtual_ones=4, virtual_zeros=4)

    # Plot the prior distribution before seeing any data at all
    x = np.linspace(0, 1, num=2**8)
    plt.title(r'Prior distribution for $\mu$')
    plt.plot(x, model.mu.pdf(x), label='Prior distribution')

    # Generate coin tosses
    num_batches = 5
    coins_per_batch = 10

    for c in range(1, num_batches + 1):

        # Draw some coins which are not fair, controlled by weights
        coins = random.choices([0, 1], weights=[0.3, 0.7], k=coins_per_batch)
        model.fit(coins)

        # Create a plot and print information
        plt.title(r'Posterior distribution for $\mu$')
        plt.plot(x, model.mu.pdf(x), alpha=c / num_batches,
                 color='red', label=f'{c} batches')
        print(f'After batch {c}, the probability of head is {model.p.mean()}')

    # Finally, show the plot
    plt.legend()
    plt.show()
