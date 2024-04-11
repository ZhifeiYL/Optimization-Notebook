"""
A custom object to implement the surrogate model "polynomial-chaos-based stochastic Kriging"
Zhifei Yuliu
zyuliu@udel.edu

Reference for SK: Staum, J. (2009b). Better Simulation Metamodeling: The why, what, and how of Stochastic Kriging.
Proceedings of the 2009 Winter Simulation Conference (WSC). https://doi.org/10.1109/wsc.2009.5429320


"""

import numpy as np
from scipy.optimize import minimize
import chaospy as cp


__all__ = ["SimpleStochasticKriging"]


class StochasticKrigingBase(object):
    def __init__(self):
        self.sigma_noise = 1e-3
        self._kernel_pars = None
        self._kernel = None
        self._trend_pars = None
        self._trend = None
        self._pars = None

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, kernel_func):
        """
        Set the kernel function.
        Expected behavior of kernel_func: kernel_func(x1, x2, pars) -> # Add dimension of output here
        """
        if not callable(kernel_func):
            raise ValueError("The kernel function must be callable.")
        self._kernel = kernel_func

    @property
    def kernel_pars(self):
        return self._kernel_pars

    @kernel_pars.setter
    def kernel_pars(self, pars):
        self._kernel_pars = pars

    @property
    def pars(self):
        # a list of parameters for hyperparameter optimization
        return self._pars

    @pars.setter
    def pars(self, pars):
        # a list of parameters for hyperparameter optimization
        self._pars = pars

    def get_pars(self):
        # retrieve the values of self._pars
        return self._pars

    @property
    def trend(self):
        return self._trend

    def train(self, X_train, Y_train, V_train):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def predict(self, X_data):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def obj_func(self, hyperpars):
        raise NotImplementedError("This method should be implemented in a subclass.")

    # Add any other utility functions here
    # For example, a method for setting the trend
    def set_trend(self, trend_func, trend_pars=None):
        self._trend = trend_func
        self._trend_pars = trend_pars


class SimpleStochasticKriging(StochasticKrigingBase):
    def set_trend(self, trend_func, trend_pars=None):
        raise NotImplementedError("Cannot set trend for simple stochastic Kriging")


    def train(self, X_train, Y_train, V_train, bounds=[(1e-5, None), (1e-5, None), (1e-5, None)]):
        """
        :param X_train: variables at design points [dim: k x d]
        :param Y_train: response(mean) [dim: k x 1]
        :param V_train: variance at each design points [dim: k x d]
        k: number of design points
        d: dimension of design variables
        """
        # dimension check:
        k, d = X_train.shape[0]

        if Y_train.shape != (k, 1):
            raise ValueError("Output vector and design point matrix must have the same number of rows.")
        if V_train.shape != (k, 1):
            raise ValueError("The variance vector and design point matrix must have the same number of rows.")
        # minX = np.min(X_train, axis=0)
        # maxX = np.max(X_train, axis=0)
        # X_normalized = (X_train - minX) / (maxX - minX)

        self.X_train = X_train
        self.k = k
        self.d = d

        # compute the intrinsic covariance matrix
        self.cov_in = np.diag(V_train)

        # compute the discrepancy/residual
        self.y_diff = Y_train - self.trend(X_train)

        # hyper-pars optimization, show progress here
        res = minimize(self.obj_func, self.get_pars(), bounds=bounds, method='L-BFGS-B')
        self.pars = res.x

        # solve system with the optimal hyperparameters
        self.solve_system()
        # compute weight W (Q in the reference)
        self.W = np.linalg.solve(self.L.T, self.Z)

    def solve_system(self):
        # compute the extrinsic covariance matrix
        self.cov_ex = self.kernel(self.X_train, self.X_train, self.kernel_pars)

        # compute the covariance matrix cov_mat
        self.cov_mat = self.cov_ex + self.cov_in

        # compute the lower triangular Cholesky factor: L
        self.L = np.linalg.cholesky(self.cov_mat)

        # compute Z
        self.Z = np.linalg.solve(self.L, self.y_diff)

    def obj_func(self, hyperpars):
        self.solve_system()
        # return the negative of profile log-likelihood  (since the goal is to maximize the "profile log-likelihood")
        return 0.5 * self.k * np.log(2*np.pi) + 2 * np.sum(np.log(np.diag(self.L))) + 0.5 * np.dot(self.Z.T, self.Z)

    def predict(self, X_data):
        vec_var = self.kernel(self.X_train, X_data, self.kernel_pars)
        return self.trend(X_data) + np.dot(vec_var.T, self.W)















