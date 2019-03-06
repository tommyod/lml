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




"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClusterMixin
from scipy import stats
import itertools
from sklearn.utils import check_array
import numbers


def generate_data(N, k, dim=6, seed=123):
    """
    Ancestral sampling to generate data.
    """

    # Set a seet for reproducible results
    np.random.seed(seed)

    # Generate mixing coefficients. They are in the range [0, 1] and sum to 1.
    pi_ks = np.random.rand(k)
    pi_ks = pi_ks / np.sum(pi_ks)
    assert np.allclose(np.sum(pi_ks), 1)

    # Generate mean vectors u_k, store these in a list.
    # Each u_k has entries in the range [0, 1].
    u_ks = [np.random.rand(dim) for j in range(k)]
    assert all(np.all((u_k <= 1) & (u_k >= 0)) for u_k in u_ks)

    # Draw mixing coefficients for each row
    mix_coeff_indices = np.random.choice(
        a=list(range(k)), size=N, replace=True, p=pi_ks
    )

    # Every row gets a mean vector u_k (prototype) based on the drawn mixing coefficient
    data = np.array(u_ks)[mix_coeff_indices, :]

    # Draw 0 or 1 based on probabilities. This vectorized operation is ~10 times
    # faster than looping over the dataset row by row.
    drawn = np.random.rand(N, dim)
    data = np.where(drawn > data, 1, 0)
    return pi_ks, u_ks, data


class LCA(BaseEstimator, ClusterMixin):
    
    def __init__(self, n_components=None, tol=0.01, max_iter=1000, verbose=0):
        """Latent Component Analysis (LCA)
        
        This implementation is based on Section 9.3.3. in Bishop.

        Parameters
        ----------
        n_components : int | None
            The number of mixture components, i.e. K in Equation (9.47) in Bishop.
        tol : float
            Stopping tolerance for EM algorithm.
        max_iter : int
            Maximum number of iterations.
        verbose : int
            The verbosity of printing. Higher means more prints.

        Attributes
        ----------
        components_ : array, [n_components, n_features]
            Components with maximum variance.
        loglike_ : list, [n_iterations]
            The log likelihood at each iteration.
        noise_variance_ : array, shape=(n_features,)
            The estimated noise variance for each feature.
        n_iter_ : int
            Number of iterations run.
        verbose : int
            Number of iterations run.
            
        Examples
        --------
        >>> X = np.array([[0, 1], [0, 1], [1, 1]])
        >>> lca_model = LCA(n_components=2)
        >>> lca_model = lca_model.fit(X)
        
        References
        ----------
        .. Christopher M. Bishop: Pattern Recognition and Machine Learning,
            Chapter 9.3.3
        """
        # Sanity checking of the inputs
        assert 0 <= tol <= 1
        assert max_iter > 0
        assert isinstance(max_iter, numbers.Integral)
        assert isinstance(max_iter, numbers.Integral)
        
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X):
        """
        Fit the model to the data.
        """
        
        # Check that the input is a non-empty 2D array containing only finite values.
        X = check_array(X, dtype=np.int)
        
        self.X = X
        n_samples, n_features = X.shape
        self.N, n_features = X.shape
        
        if self.n_components is None:
            self.n_components = n_features
            
        # Mixing coefficients are set to approximately the same value, and sum to 1
        self.pi = np.abs(np.random.randn(self.n_components)) + 100
        self.pi_ks = self.pi / np.sum(self.pi)
        
        # The cluster means (cluster prototypes) are set to the occurences
        self.u_ks = np.mean(self.X, axis=0)
        self.u_ks = [self.u_ks for i in range(self.n_components)]


        self.loglike_ = []
        for iteration in range(self.max_iter):

            mix_comp_matrix, sum_mix_comp = self._compute_mixture_components()
            log_likelihood = self._eval_log_likelihood(sum_mix_comp)
            # print(f"Iternation number {iteration}. log_likelihood is {log_likelihood}")

            resp = self._eval_expectation_step(mix_comp_matrix, sum_mix_comp)
            self.pi_ks, self.u_ks = self._eval_maximization_step_naive(resp)

        # print("Fitted pi_k:", self.pi_ks)
        # print("Fitted u_ks:", self.u_ks)
        
        return self

    def _compute_mixture_components(self):
        """
        Computes the mixture components.
        
        The returned matrix has size (N, K).
        
        The (n, k) entry consists of
        pi_k * p(x_n | mu_k)
        
        This computation is needed for the log-likelihood, and for the E-step.
        """
        mixture_components = np.empty((self.N, self.n_components))

        for k in range(self.n_components):

            u_k_repeated = np.repeat(self.u_ks[k].reshape(1, -1), self.N, axis=0)
            # prob_X_uk = stats.bernoulli.pmf(k=self.X, p=u_k_repeated)
            prob_X_uk = np.where(self.X == 1, u_k_repeated, 1 - u_k_repeated)
            # assert np.allclose(prob_X_uk, prob_X_uk2)
            # u_k_repeated**self.X
            prob_X_uk = np.product(prob_X_uk, axis=1)
            prob_X_uk = prob_X_uk * self.pi_ks[k]

            mixture_components[:, k] = prob_X_uk

        return mixture_components, np.sum(mixture_components, axis=1)

    def _eval_log_likelihood_naive(self):
        """
        Evaluate the log-likelihood function.
        """

        log_likelihood = 0
        for n in range(self.N):
            inside_log = 0
            for k in range(self.n_components):
                bernoulli = stats.bernoulli.pmf(k=self.X[n, :], p=self.u_ks[k])
                inside_log += self.pi_ks[k] * np.product(bernoulli)

            log_likelihood += np.log(inside_log)
        return log_likelihood

    def _eval_log_likelihood(self, sum_mix_comp):
        """
        Evaluate the log-likelihood function.
        
        Evaluates Equation (9.51) in Bishop.
        """

        # Sum over k
        temp = sum_mix_comp
        assert len(temp) == self.N

        # Take logarithms
        temp = np.log(temp)

        # Sum over N and return
        ans = np.sum(temp)
        # assert np.allclose(ans, self._eval_log_likelihood_naive())
        return ans

    def _eval_expectation_step_naive(self):

        responsibilities = np.empty((self.N, self.n_components))

        for n in range(self.N):
            for k in range(self.n_components):
                bernoulli = stats.bernoulli.pmf(k=self.X[n, :], p=self.u_ks[k])
                responsibilities[n, k] = self.pi_ks[k] * np.product(bernoulli)

        # Vectorized division
        responsibilities /= np.sum(responsibilities, axis=1).reshape(-1, 1)

        assert np.allclose(np.ones_like(self.N), np.sum(responsibilities, axis=1))
        return responsibilities

    def _eval_expectation_step(self, mix_comp_matrix, sum_mix_comp):

        # Vectorized division
        responsibilities = mix_comp_matrix / sum_mix_comp.reshape(-1, 1)

        assert np.allclose(np.ones_like(self.N), np.sum(responsibilities, axis=1))
        # assert np.allclose(responsibilities, self._eval_expectation_step_naive())

        return responsibilities

    def _eval_maximization_step_naive(self, responsibilities):

        N_ks = np.sum(responsibilities, axis=0)
        new_pi_ks = N_ks / self.N
        # print(new_pi_ks)
        # assert np.allclose(np.sum(new_pi_ks), 1)

        new_u_ks = []
        for k in range(self.n_components):
            u_k = np.sum(responsibilities[:, [k]] * self.X, axis=0)
            new_u_ks.append(u_k / N_ks[k])

        return new_pi_ks, new_u_ks

    def _eval_maximization_step(self):
        pass

    def predict(self, X):
        pass
    
    
def test_vs_naive():
    
    generator = enumerate(itertools.product([10, 100, 1000], [1, 2, 3], [2, 3, 4]), 1)
    for test_num, (N, k, dim) in generator:
        print('Running test {}'.format(test_num))
        _, _, data = generate_data(N=N, k=k, dim=dim, seed=test_num)
        
        lca_model = LCA(n_components=k)
        lca_model
        


pi_ks, u_ks, data = generate_data(N=10000, k=2, dim=3)

print(data)


lca_model = LCA(n_components=2)

lca_model.fit(data)

print("Data pi_k:", pi_ks)
print("Data u_ks:", u_ks)

# %timeit lca_model._eval_log_likelihood() # (N=1000, k=3, dim=5)
# 203 ms ± 3.54 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 1.92 ms ± 10.6 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


if __name__ == '__main__':
    test_vs_naive()
    
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v"])
