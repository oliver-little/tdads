#This sklearn import statement below is causing this issue:
#ImportError: DLL load failed: The specified module could not be found.
from sklearn.model_selection import GridSearchCV

'''Optimisation of hyperparameters for feed forward neural network'''

#Fine tuning epochs number
def optimise_epochs(x_train, y_train, model):
    epochs = [10, 15, 20]
    param_grid = dict(epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(x_train, y_train)
    # Summarise results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

#Fine tuning optimiser
def optimise_optimizer(x_train, y_train, model):
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam', 'Ftrl']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
    grid_result = grid.fit(x_train, y_train)
    # Summarise results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

#Fine tuning learning rate
def optimise_lrate(x_train, y_train, model):
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    param_grid = dict(learn_rate=learn_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
    grid_result = grid.fit(x_train, y_train)
    # Summarise results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

#Fine tuning activation function
def optimise_activation(x_train, y_train, model):
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    param_grid = dict(activation=activation)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
    grid_result = grid.fit(x_train, y_train)
    # Summarise results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))