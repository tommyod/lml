#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 16:50:43 2018

@author: tommy
"""
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator


class BayesianGaussian1DKnownVar(BaseEstimator):
    """
    Bayesian interference of a Gaussian with known variance.
    The variance supplied is used in the prior distribution over mu,
    as well as in the likelihood function. This corresponds with Figure 2.12.

    See section 2.3.6 in Bishop.

    Examples
    --------
    >>> model = BayesianGaussian1DKnownVar(mean_0=0, sigma_sq_0=1)
    >>> model.fit([0.9,  1.7,  1.1, -0.3,  1.2])
    >>> model.mu.mean()
    0.7666666666666668
    >>> model.fit([1 for i in range(1000)])
    >>> abs(model.mu.mean() - 1) < 0.01
    True
    """

    def __init__(self, mean_0, sigma_sq_0):
        """
        Initilize a Gaussian model, with known variance and p(mu) ~ Normal.


        """
        self.mean_0 = mean_0
        self.sigma_sq_0 = sigma_sq_0

        # While the quantities above will be updated, the one below is constant
        # In equations (2.141) and (2.142) we need a variance of the observed
        # data (denoted sigma^2), and this is not actually computed from the
        # data - instead self.sigma_sq_likelihood is used
        self.var = sigma_sq_0

    def fit(self, x_data):
        """
        Fit the model to data, i.e. update the posterior distribution.
        """
        mean = np.mean(x_data)
        N = len(x_data)

        # Equation (2.141) in Bishop
        denom = N * self.sigma_sq_0 + self.var
        self.mean_0 = ((self.var / denom) * self.mean_0 +
                       (N * self.sigma_sq_0 / denom) * mean)

        # Equation (2.142) in Bishop
        self.sigma_sq_0 = (1 / ((1 / self.sigma_sq_0) +
                                (N / self.var)))

    @property
    def mu(self):
        """
        Return the pdf over mu.
        """
        return stats.norm(loc=self.mean_0, scale=np.sqrt(self.sigma_sq_0))


if __name__ == "__main__":
    import pytest
    pytest.main(args=[__file__, '--doctest-modules', '-v', '--capture=sys',
                      '--disable-warnings'])

if __name__ == '__main__':

    # Import packages
    import matplotlib.pyplot as plt

    # Create a model for coin tosses, see Section 2.1 in Bishop
    model = BayesianGaussian1DKnownVar(mean_0=0, sigma_sq_0=0.1)

    # Plot the prior distribution before seeing any data at all
    x = np.linspace(-1.5, 1.5, num=2**8)
    plt.title(r'Prior distribution for $\mu$')
    plt.plot(x, model.mu.pdf(x), label='Prior distribution')

    N = 1
    generating_dist = stats.norm(loc=0.8, scale=np.sqrt(0.1))
    x_data = generating_dist.rvs(N, random_state=42)
    plt.scatter(x_data, np.zeros_like(x_data), marker='|', c='r', label='Data')
    model.fit(x_data)

    plt.plot(x, model.mu.pdf(x), label='Posterior distribution N = 1')

    N = 9
    x_data = generating_dist.rvs(N, random_state=42)
    plt.scatter(x_data, np.zeros_like(x_data), marker='|', c='r', label='Data')
    model.fit(x_data)
    plt.plot(x, model.mu.pdf(x), label='Posterior distribution N = 10')

    # Finally, show the plot
    plt.legend()
    plt.show()
