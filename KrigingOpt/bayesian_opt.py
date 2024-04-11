import numpy as np
from scipy.optimize import minimize

from surrogate import Surrogate


class Simulator:
    def __init__(self):
        self._evaluate = None

    @property
    def evaluate(self):
        return self._evaluate

    @evaluate.setter
    def evaluate(self, func):
        # func(x, **kwargs) -> y, v
        # x: [n x d] array where n is the number of design points and d is the dimension.
        # repetition=1 (default keyword arg) , m = repetition
        # y: [n x m] response at each design point
        if not callable(func):
            raise ValueError("The evaluate property must be a callable function.")
        self._evaluate = func


class AcquisitionFunction:
    def __init__(self):
        pass

    def evaluate(self, x, surrogate, current_best):
        # Define the acquisition function
        raise NotImplementedError("This method should be implemented in a subclass.")


class Optimizer:
    def __init__(self, simulator, surrogate, acquisition_function):
        self.simulator = simulator
        self.surrogate = surrogate
        self.acquisition_function = acquisition_function
        self.current_best = None

    def optimize(self, initial_points, n_iterations):
        # Perform optimization

        for iteration in range(n_iterations):
            # Select the next point to evaluate
            x_next = self.select_next_point()

            # Evaluate the simulator
            obj, constr = self.simulator.evaluate(x_next)

            # Update the surrogate model
            # ...

            # Update the current best if necessary
            # ...

        # return self.current_best

        raise NotImplementedError("This method should be implemented in a subclass.")

    def select_next_point(self):
        # Use the acquisition function to select the next point x: [n x d]
        raise NotImplementedError("This method should be implemented in a subclass.")