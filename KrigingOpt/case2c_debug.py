import numpy as np
import chaospy as cp
import GPy
from sklearn.linear_model import LarsCV, LassoLarsCV

from stochastic_kriging import SimpleStochasticKriging, UK


# Data Generation
np.random.seed(1234)

n = 50
num_rep = 20
X = np.linspace(0,10,n)
X = X.reshape(-1,1)
sigma = 0.5

f_true = lambda x: -np.cos(np.pi * x) + np.sin(4*np.pi * x)

Y = np.empty((n, num_rep))
for i in range(n):
    Y[i, :] = f_true(X[i]) + sigma * np.random.normal(loc=0, scale=sigma, size=num_rep)

X_train = X
Y_train = np.mean(Y, axis=1, keepdims=True)
V_train = np.var(Y, axis=1, keepdims=True)

gp_model = GPy.models.GPRegression(X, Y, GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.))
gp_model.optimize(messages=False)

def rbf_kernel(x1, x2, pars):
    sigma_f, length_scale = pars
    sqdist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f ** 2 * np.exp(-0.5 / length_scale ** 2 * sqdist)

# sk construction
sk = SimpleStochasticKriging()
sk.kernel = rbf_kernel
sk.train(X_train, Y_train, V_train)

# pck construction

def optimize_pce(pce_expansion, data):
    pce_x, pce_y = data
    lars = LarsCV(fit_intercept=False, max_iter=7)
    pce, coeffs = cp.fit_regression(
        pce_expansion, pce_x, pce_y, model=lars, retall=True)
    pce_expansion_ = pce_expansion[coeffs != 0]
    return pce

pcsk = UK()
distribution = cp.Uniform(0, 10)
expansion = cp.generate_expansion(20, distribution, normed=True)
pcsk.set_trend(expansion)
pcsk.fit_trend(optimize_pce, [X_train.reshape(-1,), Y_train.reshape(-1,)])
pcsk.kernel = rbf_kernel
pcsk.train(X_train, Y_train, V_train)

# First try to plot prediction of the mean, mse with the true function

def calculate_ermse(y_true, y_pred):
    """
    Calculate Empirical Root Mean Squared Error (ERMSE).
    """
    ermse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return ermse


def calculate_nmae(y_true, y_pred):
    """
    Calculate Normalized Maximum Absolute Error (NMAE).
    """
    sigma_ys = np.sqrt(np.mean((y_pred - y_true) ** 2))
    nmae = np.max(np.abs(y_pred - y_true)) / sigma_ys
    return nmae


all_models = dict()
all_models["Kriging_"+str(100) + "_" + str(10)] = gp_model
all_models["SK_"+str(100) + "_" + str(10)] = sk
all_models["PCSK_"+str(100) + "_" + str(10)] = pcsk



def perform_analysis(n_validate, metric_functions, f_true, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Generate validation data
    # seed the random x's, if None don't seed
    X_validate = np.linspace(0,10,n_validate)
    X_validate = X_validate.reshape(-1,1)
    Y_validate = f_true(X_validate)

    # Dictionary to store predictions and errors
    predictions = {}
    metrics_set = [dict() for _ in range(len(metric_functions))]

    # Iterate over each model and evaluate
    for n, model in all_models.items():
        # Predict using the model
        Y_pred, Y_var = model.predict(X_validate)

        # Calculate the error using the provided metric function
        metrics = [metric_function(Y_validate, Y_pred) for metric_function in metric_functions]

        # Store results
        predictions[n] = Y_pred
        for i in range(len(metrics)):
            metrics_set[i][n] = metrics[i]

    return X_validate, predictions, metrics_set

# 100 trails for replications
# 1e5 points for each validation trial
num_seeds = 100
seeds = []
n_validation = int(1e5)

# Store the results for each seed
all_metrics1 = {}
all_metrics2 = {}
all_predictions = {}
all_X_validations = {}

# Example for ermse
for i in range(num_seeds):
    seed = np.random.randint(0, 100000)  # Generate a random seed
    seeds.append(seed)
    X_validation, predictions, metrics = perform_analysis(n_validation, [calculate_ermse, calculate_nmae], f_true, seed=seed)

    # Store the results
    all_metrics1[seed] = metrics[0]
    all_metrics2[seed] = metrics[1]
    all_predictions[seed] = predictions
    all_X_validations[seed] = X_validation

# scenarios: keys to all models
scenarios = all_models.keys()

data_runs = [[] for _ in range(len(scenarios))]  # Adjust 'len(scenarios)' to the actual number of scenarios you have

for metrics in all_metrics1.values():
    for i, n in enumerate(scenarios):
        data_runs[i].append(metrics[n])

1 == 1