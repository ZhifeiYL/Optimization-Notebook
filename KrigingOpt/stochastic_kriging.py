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

from surrogate import Surrogate

__all__ = ["SimpleStochasticKriging"]


class StochasticKrigingBase(Surrogate):
    def __init__(self):
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
        # Retrieve the current values of the kernel parameters
        return [getattr(self, par) for par in self._kernel_pars]

    @kernel_pars.setter
    def kernel_pars(self, pars_values):
        # Update the values of the parameters
        for par, val in zip(self._kernel_pars, pars_values):
            setattr(self, par, val)

    @property
    def pars(self):
        # Retrieve the current values of the parameters listed in _pars
        return [getattr(self, par) for par in self._pars]

    @pars.setter
    def pars(self, pars_values):
        # Update the values of the parameters
        for par, val in zip(self._pars, pars_values):
            setattr(self, par, val)

    @property
    def trend(self):
        return self._trend

    def trend_eval(self, x_data):
        return self.trend(x_data)

    def train(self, X_train, Y_train, V_train):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def predict(self, X_data):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def obj_func(self, hyperpars):
        raise NotImplementedError("This method should be implemented in a subclass.")

    # For example, a method for setting the trend
    def set_trend(self, trend_func, trend_pars=None):
        self._trend = trend_func
        self._trend_pars = trend_pars

    def fit_trend(self, optimize_trend, data):
        raise NotImplementedError("This method should be implemented in a subclass.")


class SimpleStochasticKriging(StochasticKrigingBase):
    def __init__(self, mean=0):
        super().__init__()
        self._trend = lambda x: mean   # zero mean
        self.sigma_noise = 1e-3     # pars
        self.sigma_f = 1e-3     # kernel par 1
        self.length_scale = 1e-3    # kernel par 2
        self._pars = ['sigma_noise', 'sigma_f', 'length_scale']
        self._kernel_pars = ['sigma_f', 'length_scale']

    def set_trend(self, trend_func, trend_pars=None):
        raise NotImplementedError("Cannot set trend for simple stochastic Kriging")


    def train(self, X_train, Y_train, V_train, bounds=[(1e-5, None), (1e-5, None), (1e-5, None)], silent=False):
        """
        :param X_train: variables at design points [dim: k x d]
        :param Y_train: response(mean) [dim: k x 1]
        :param V_train: variance at each design points [dim: k x 1]
        k: number of design points
        d: dimension of design variables
        """
        # dimension check:
        k, d = X_train.shape

        if Y_train.shape != (k, d):
            raise ValueError("Output vector and design point matrix must have the same number of rows.")
        if V_train.shape != (k, 1):
            raise ValueError("The variance vector and design point matrix must have the same number of rows.")
        # minX = np.min(X_train, axis=0)
        # maxX = np.max(X_train, axis=0)
        # X_normalized = (X_train - minX) / (maxX - minX)

        self.X_train = X_train
        self.k = k
        self.d = d

        # compute the intrinsic covariance matrix [k x k] from V_train
        self.cov_in = np.diag(V_train.flatten())

        # compute the discrepancy/residual
        self.y_diff = Y_train - self.trend_eval(X_train)

        # hyper-pars optimization, show progress+result here if not silent
        options = {'disp': None if silent else True}
        res = minimize(self.obj_func, np.array(self.pars), bounds=bounds, method='L-BFGS-B', options=options)
        #  check convergence status

        self.pars = res.x

        # solve system with the optimal hyperparameters
        self.solve_system()
        # compute weight W (Q in the reference)
        self.W = np.linalg.solve(self.L.T, self.Z)

        # compute process variance and inverse of covariance matrix
        self.process_var = V_train  # process variance [k, 1]
        self.cov_mat_inv = np.linalg.inv(self.cov_mat)      # inverse of covariance matrix [k x k]

    def solve_system(self):
        # compute the extrinsic covariance matrix
        if self.kernel is None:
            raise NotImplementedError("No kernel defined.")
        self.cov_ex = self.kernel(self.X_train, self.X_train, self.kernel_pars)

        # compute the covariance matrix cov_mat
        self.cov_mat = self.cov_ex + self.cov_in

        # compute the lower triangular Cholesky factor: L
        self.L = np.linalg.cholesky(self.cov_mat)

        # compute Z
        self.Z = np.linalg.solve(self.L, self.y_diff)

    def obj_func(self, hyperpars):
        self.pars = hyperpars
        self.solve_system()
        # return the negative of profile log-likelihood  (since the goal is to maximize the "profile log-likelihood")
        return 0.5 * self.k * np.log(2*np.pi) + 2 * np.sum(np.log(np.diag(self.L))) + 0.5 * np.dot(self.Z.T, self.Z)

    def predict(self, X_data):
        vec_var = self.kernel(self.X_train, X_data, self.kernel_pars)   # [k x num_data]
        predicted_mean = self.trend_eval(X_data) + np.dot(vec_var.T, self.W)     # [num_data x 1]

        num_data = X_data.shape[0]  # Number of prediction points

        y_diff_pred = 1 - np.sum(np.dot(self.cov_mat.T, self.cov_mat_inv) * self.cov_mat.T, axis=1,
                                 keepdims=True)  # [k x 1]
        # MSE calculation [num_data x 1]
        mse = np.zeros((num_data, 1))
        # to be implemented
        return predicted_mean, mse


class UK(SimpleStochasticKriging):
    def __init__(self):
        self._trend = None  # set it later
        self.sigma_noise = 1e-3  # pars
        self.sigma_f = 1e-3  # kernel par 1
        self.length_scale = 1e-3  # kernel par 2
        self._pars = ['sigma_noise', 'sigma_f', 'length_scale']
        self._kernel_pars = ['sigma_f', 'length_scale']
        self._kernel = None
        self._trend_pars = None

    def set_trend(self, trend_func, trend_pars=None):
        self._trend = trend_func
        self._trend_pars = trend_pars

    def fit_trend(self, optimize_trend, data):
        self._trend = optimize_trend(self.trend, data)

    def trend_eval(self, x_data):
        return self.trend(x_data.reshape(-1,)).reshape(-1,1)















