#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 09:35:35 2018

@author: tommy
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class PolynomialCurve(BaseEstimator, RegressorMixin):
    """
    Fit a 1D, arbitrary degree polynomial to data points.

    This estimator minimizes the sum of squares residuals, as described in
    Equation (1.2) in Bishop. No regularization is used.
    """

    def __init__(self, degree=1):
        """
        Inititalize the estimator by specifying the degree of the polynomial.

        Parameters
        ----------
        degree : int
            The degree of the polynomial, must be >= 0. Called M in Bishop.
        """
        self.degree = degree

    def fit(self, x_data, y_data):
        """
        Fit the estimator to the 1D data.
        """
        # Create a Vandermode matrix.
        X_data = np.vander(x_data, N=self.degree + 1, increasing=True)

        # Solve the least squares problem. Solving using the Moore–Penrose
        # inverse should never be done in code, since the vandermode matrix
        # has a high conditioning number, the solution is numerically instable
        # and slower than solving using the SVD.
        w, residuals, rank, s = np.linalg.lstsq(X_data, y_data, rcond=None)
        self.w_, self.residuals_ = w, residuals
        return self

    def predict(self, x_data):
        """
        Predict on 1D data.
        """
        X_data = np.vander(x_data, N=self.degree + 1, increasing=True)
        return np.dot(X_data, self.w_)


class PolynomialCurveRidge(BaseEstimator, RegressorMixin):
    """
    Fit a 1D, arbitrary degree polynomial to data points.

    This estimator minimizes the sum of squares residuals plus a regularization
    term, as described in Equation (1.4) in Bishop.
    """

    def __init__(self, degree=1, alpha=1.0):
        """
        Inititalize the estimator by specifying the degree of the polynomial,
        and the regularization parameter alpha.

        Parameters
        ----------
        degree : int
            The degree of the polynomial, must be >= 0. Called M in Bishop.
        alpha : float
            The regularization parameter, see eqn (1.4). Must be >= 0.
        """
        self.degree = degree
        self.alpha = alpha

    def fit(self, x_data, y_data):
        """
        Fit the estimator to the 1D data.
        """
        # Create a Vandermode matrix
        X_data = np.vander(x_data, N=self.degree + 1, increasing=True)

        # The left hand side and right hand side of the equation for ridge
        # regularized least squares. Computing X.T * X is the expensive part
        # Could have used np.linalg.solve
        lhs = np.dot(X_data.T, X_data) + np.eye(self.degree + 1) * self.alpha
        rhs = np.dot(X_data.T, y_data)

        # Solve the linear equation. np.linalg.solve is faster than
        # np.linalg.lstsq, but we assume that the system of equations has
        # a unique solution here. I.e. not under- or overdetermined.
        w = np.linalg.solve(lhs, rhs)
        self.w_ = w
        return self

    def predict(self, x_data):
        """
        Predict on 1D data.
        """
        X_data = np.vander(x_data, N=self.degree + 1, increasing=True)
        return np.dot(X_data, self.w_)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    np.random.seed(42)
    # Generate some random data
    N = 11
    x_data = np.random.rand(N)
    y_data = np.sin(2 * np.pi * x_data) + np.random.randn(N) / 5

    plt.title("Polynomial curve fitting with regularization")
    plt.scatter(x_data, y_data, color="red", label=f"{N} data points")

    degree = 8
    poly = PolynomialCurve(degree=degree)
    poly.fit(x_data, y_data)
    x_smooth = np.linspace(0, 1, num=2 ** 8)
    plt.plot(x_smooth, np.sin(2 * np.pi * x_smooth), "--", label="True function")
    plt.plot(x_smooth, poly.predict(x_smooth), label=f"Degree {degree} polynomial")

    alpha = np.exp(-10)
    poly = PolynomialCurveRidge(degree=degree, alpha=alpha)
    poly.fit(x_data, y_data)
    x_smooth = np.linspace(0, 1, num=2 ** 8)
    plt.plot(
        x_smooth,
        poly.predict(x_smooth),
        label=f"Degree {degree} polynomial with regularization",
    )
    plt.legend()
    plt.show()
