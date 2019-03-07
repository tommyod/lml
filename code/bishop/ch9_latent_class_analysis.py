#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script implements Mixtures of Bernoulli distributions, also called
"latent class analysis". The implementation follows Section 9.3.3 in the book
"Pattern Recognition and Machine Learning" by Bishop.

Theory and mathematics
----------------------

There are D binary variables x_i, comprising a vector x of dimension D.
Each x_is is governed by a Bernoulli distribution, so that

p(x | u) = \prod_i u_i^{x_i} * (1 - u_i)^{1 - x_i}.         (Eqn 9.44 in Bishop)

In other words, x_i and x_j are independent, and the probability that x_i = 1 is u_i.
The mixture model is given by

p(x) = \sum_k \pi_k p(x | u_k)                              (Eqn 9.47 in Bishop)

The log likelihood is maximized with respect to \pi_k and u_k.

Compared with other implementations
-----------------------------------

Although this implementation is mean to be pedagogical, I've timed it against one other
Python implementations on a dataset with N = 10 000, 100 iterations, 5 dimensions
and 3 clusters. The results are:
    
    - 329 ms ± 8.75 ms : This implementation.
    - 10.1 s ± 825 ms : https://github.com/dasirra/latent-class-analysis

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
import itertools
from sklearn.utils import check_array
import numbers


class LCA(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, n_components=None, tol=0.01, max_iter=100, verbose=0):
        """Latent Component Analysis (LCA)
        
        This implementation is based on Section 9.3.3. in Bishop.

        Parameters
        ----------
        n_components : int | None
            The number of mixture components, i.e. K in Equation (9.47) in Bishop.
        tol : float
            Stopping tolerance for EM algorithm, absolute number.
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
        >>> lca_model = lca_model.fit(X, seed=1)
        >>> lca_model.predict_proba(np.array([[0, 1], [1, 1]]))
        array([[0.7596397 , 0.2403603 ],
               [0.00209745, 0.99790255]])
    
    
        References
        ----------
        .. Christopher M. Bishop: Pattern Recognition and Machine Learning,
            Chapter 9.3.3
        """
        # Sanity checking of the inputs
        assert tol > 0
        assert max_iter > 0
        assert isinstance(max_iter, numbers.Integral)
        assert isinstance(max_iter, numbers.Integral)

        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y=None, seed=123):
        """
        Fit the model to the data.
        """
        if not y is None:
            raise ValueError("Unsupervised algorithm. `y` must be None.")

        # Check that the input is a non-empty 2D array containing only finite values.
        assert np.array_equal(X, X.astype(bool))
        X = check_array(X, dtype=np.int)

        self.X = X
        self._n_samples, self._n_features = X.shape

        if self.n_components is None:
            self.n_components = self._n_features

        # Mixing coefficients are set to approximately the same value, and sum to 1
        np.random.seed(seed)
        self.pi_ks = np.abs(np.random.rand(self.n_components)) + 1
        self.pi_ks = self.pi_ks / np.sum(self.pi_ks)

        # The cluster means (cluster prototypes) are set to the occurences
        # self.u_ks = np.mean(self.X, axis=0)
        self.u_ks = [np.random.rand(self._n_features) for _ in range(self.n_components)]

        self.loglike_ = []
        for i in range(1, self.max_iter + 1):

            # Compute the mixture component matrix, and sum over the columns
            mix_comp_matrix = self._compute_mixture_components(self.X)
            sum_comp = np.sum(mix_comp_matrix, axis=1)

            # Compute the log likelihood at this iteration
            self.loglike_.append(self._eval_log_likelihood(sum_comp))

            if self.verbose > 0:
                print(f"Iternation number {i}. LL is {round(self.loglike_[-1], 2)}")

            responsibilities = self._eval_expectation_step(mix_comp_matrix, sum_comp)
            self.pi_ks, self.u_ks = self._eval_maximization_step(responsibilities)

            if len(self.loglike_) > 1:
                if abs(self.loglike_[-1] - self.loglike_[-2]) < 0.01:
                    if self.verbose > 0:
                        print(f"Breaking due to tolerance.")
                    break
        else:
            if self.verbose > 0:
                print(f"Breaking due to number of iterations.")

        return self

    def _compute_mixture_components(self, X):
        """
        Computes the mixture components.
        
        This computation exploits the fact that the same terms are needed for
        computing the log likelihood and the E-step of the EM algorithm.
        Equations (9.51) (log likelihood) and (9.56) (E-step) are similar enough
        to warrant a function which computes the needed quantities once.
        
        This function returns a matrix `mix_comp_matrix` of shape 
        (n_samples, n_components) in the notation of sklearn, or (N, K) in Bishops
        notation. Entry (n, k) consists of pi_k * p(x_n | mu_k).
        """
        assert np.array_equal(X, X.astype(bool))
        n_samples, _ = X.shape
        mix_comp_matrix = np.empty((n_samples, self.n_components))

        for k in range(self.n_components):
            u_k_repeated = np.tile(self.u_ks[k], (n_samples, 1))

            # The lines below are equivalent, but the second is much faster
            # prob_X_uk_slow = stats.bernoulli.pmf(k=self.X, p=u_k_repeated)
            prob_X_uk = np.where(X == 1, u_k_repeated, 1 - u_k_repeated)
            # assert np.allclose(prob_X_uk, prob_X_uk_slow)

            mix_comp_matrix[:, k] = np.product(prob_X_uk, axis=1) * self.pi_ks[k]

        return mix_comp_matrix

    def _eval_log_likelihood(self, sum_mix_comp):
        """
        Evaluate the log-likelihood function.
        """
        # This line evalutes Equation (9.51) in bishop, given the N inner sums
        # as a vector, here denoted by 'sum_mix_comp'
        return np.sum(np.log(sum_mix_comp))

    def _eval_expectation_step(self, mix_comp_matrix, sum_mix_comp):
        """
        The (E)xpectation step of the EM algorithm. Computes the responsibilities.
        """
        # This is the division in Equation (9.56) in Bishop
        return mix_comp_matrix / sum_mix_comp.reshape(-1, 1)

    def _eval_maximization_step(self, responsibilities):
        """
        The (M)aximization step of the EM algorithm.
        """
        # These values are used twice, so only compute them once
        N_ks = np.sum(responsibilities, axis=0)
        # This is Eqn (9.60) in Bishop
        new_pi_ks = N_ks / self._n_samples

        # This is Eqn (9.59) in Bishop
        new_u_ks = []
        for k in range(self.n_components):
            u_k = np.sum(responsibilities[:, [k]] * self.X, axis=0)
            new_u_ks.append(u_k / N_ks[k])

        return new_pi_ks, new_u_ks

    def predict(self, X):
        """
        Predict a the most likely latent variable.
        """
        # Compute the mixture component matrix. This is argmax of eqn (9.56) in Bishop
        return np.argmax(self._compute_mixture_components(X), axis=1)

    def predict_proba(self, X):
        """
        Predicts probabilities of samples belonging to each latent variable.
        """
        # Computes the probabilities of Eqn (9.56) in bishop
        mix_comp_matrix = self._compute_mixture_components(X)
        sum_mix_comp = np.sum(mix_comp_matrix, axis=1)
        return self._eval_expectation_step(mix_comp_matrix, sum_mix_comp)


# ------------------------------------------------------------------
# ----------------- FUNCTIONS FOR TESTING AND EXAMPLES -------------
# ------------------------------------------------------------------


def generate_data(N, k, dim=6, seed=123):
    """
    Ancestral sampling to generate data. The data is generate by first selecting one
    of k distributions, then by bernoulli.
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
    return pi_ks, u_ks, mix_coeff_indices, data


def test_vs_naive():

    generator = enumerate(itertools.product([10, 100, 1000], [2, 3], [3, 4]), 1)
    for test_num, (N, k, dim) in generator:
        print("Running test {}".format(test_num))
        pi_ks, u_ks, latent_z, data = generate_data(N=N, k=k, dim=dim, seed=test_num)
        lca_model = LCA(n_components=k)

        lca_model.fit(data, seed=1)
        plt.title(f"N={N}, k={k}, dim={dim}")
        plt.plot(lca_model.loglike_)
        plt.grid(True)

        lca_model.predict(data[:10, :])

        print("Data pi_k:", pi_ks)
        print("Data u_ks:", u_ks)
        plt.show()
        print()

    print()


if __name__ == "__main__":
    test_vs_naive()

    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[__file__, "--doctest-modules", "-v"])
