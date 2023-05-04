import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from modAL.models import BayesianOptimizer
from modAL.acquisition import max_EI, max_PI, max_UCB

#%% First sample

X = np.linspace(0, 1, 1000).reshape(-1, 1)

X_initial = np.array([[0],[0.1], [0.25], [0.5], [0.75] ,[1]])
y_initial = np.array([[0.2821],[0.3524],[0.412],[0.43] , [0.29] , [0.2851]])

# defining the kernel for the Gaussian process
kernel = Matern(length_scale=1.0)
regressor = GaussianProcessRegressor(kernel=kernel)

optimizer = BayesianOptimizer(
    estimator=regressor,
    X_training=X_initial, y_training=y_initial,
    query_strategy=max_EI #max_EI, max_PI, max-UCB
)

query_idx, query_inst = optimizer.query(X)

print("Next sample to be measured: ", query_inst)

## plotting
y_pred, y_std = optimizer.predict(X, return_std=True)
y_pred, y_std = y_pred.ravel(), y_std.ravel()
X_max, y_max = optimizer.get_max()

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 5))
    plt.scatter(optimizer.X_training, optimizer.y_training, c='k', s=50, label='Observed Samples')
    plt.axvline(x = query_inst, color = 'r', label = 'Next Sample (query): x ={}'.format(np.round(*query_inst[0],8)))
    plt.plot(X.ravel(), y_pred, label='GP regressor')
    plt.fill_between(X.ravel(), y_pred - y_std, y_pred + y_std, alpha=0.5)
    plt.title('Initial GP with first selected query')
    plt.legend()
    plt.savefig('1.jpg', dpi = 300)
    plt.show()

#%% Second Sample

X = np.linspace(0, 1, 1000).reshape(-1, 1)

X_initial = np.array([[0],[0.1], [0.25],[0.40840841], [0.5], [0.75] ,[1]])
y_initial = np.array([[0.2821],[0.3524],[0.412], [0.4432] ,[0.43] , [0.29] , [0.2851]])

# defining the kernel for the Gaussian process
kernel = Matern(length_scale=1.0)
regressor = GaussianProcessRegressor(kernel=kernel)

optimizer = BayesianOptimizer(
    estimator=regressor,
    X_training=X_initial, y_training=y_initial,
    query_strategy=max_EI #max_EI, max_PI, max-UCB
)

query_idx, query_inst = optimizer.query(X)

print("Next sample to be measured: ", query_inst)

## plotting
y_pred, y_std = optimizer.predict(X, return_std=True)
y_pred, y_std = y_pred.ravel(), y_std.ravel()
X_max, y_max = optimizer.get_max()

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 5))
    plt.scatter(optimizer.X_training, optimizer.y_training, c='k', s=50, label='Observed Samples')
    plt.axvline(x = query_inst, color = 'r', label = 'Next Sample (query): x ={}'.format(np.round(*query_inst[0],8)))
    plt.plot(X.ravel(), y_pred, label='GP regressor')
    plt.fill_between(X.ravel(), y_pred - y_std, y_pred + y_std, alpha=0.5)
    plt.title('Updated GP with second selected query')
    plt.legend()
    plt.savefig('2.jpg', dpi = 300)
    plt.show()

#%% Final GP

X = np.linspace(0, 1, 1000).reshape(-1, 1)

X_initial = np.array([[0],[0.1], [0.25],[0.40840841],[0.43543544], [0.5], [0.75] ,[1]])
y_initial = np.array([[0.2821],[0.3524],[0.412], [0.4432] , [0.4117],[0.43] , [0.29] , [0.2851]])

# defining the kernel for the Gaussian process
kernel = Matern(length_scale=1.0)
regressor = GaussianProcessRegressor(kernel=kernel)

optimizer = BayesianOptimizer(
    estimator=regressor,
    X_training=X_initial, y_training=y_initial,
    query_strategy=max_EI #max_EI, max_PI, max-UCB
)

query_idx, query_inst = optimizer.query(X)

print("Next sample to be measured: ", query_inst)

## plotting
y_pred, y_std = optimizer.predict(X, return_std=True)
y_pred, y_std = y_pred.ravel(), y_std.ravel()
X_max, y_max = optimizer.get_max()

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 5))
    plt.scatter(optimizer.X_training, optimizer.y_training, c='k', s=50, label='Observed Samples')
    plt.scatter(X_max, y_max, c='r', s=70, label='Best Observed Sample')
    #plt.axvline(x = query_inst, color = 'r', label = 'Next Sample (query): x ={}'.format(np.round(*query_inst[0],8)))
    plt.plot(X.ravel(), y_pred, label='GP regressor')
    plt.fill_between(X.ravel(), y_pred - y_std, y_pred + y_std, alpha=0.5)
    plt.title('Final GP')
    plt.legend()
    plt.savefig('3.jpg', dpi = 300)
    plt.show()
