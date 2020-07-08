import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers.core import *
from tensorflow.python.keras.optimizers import *
import tensorflow as tf
from keras.utils import plot_model
#This import statement below is causing issues:
#ImportError: DLL load failed: The specified module could not be found.
#from sklearn.model_selection import GridSearchCV


#from tensorboard.plugins.hparams import api as hp

'''A simple feed forward neural network with two hidden layers.'''

mnist = tf.keras.datasets.mnist

class_names = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#Display the first 10 images
def display_data(x_train, y_train):
    num = 10
    images = x_train[:num]
    labels = y_train[:num]
    num_row = 2
    num_col = 5

    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col, 2*num_row))
    plt.title('Images in Training Set')
    for i in range(num):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(images[i], cmap='gray_r')
        ax.set_title('Label: {}'.format(labels[i]))
    plt.show()

#Data preprocessing

def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    display_data(x_train, y_train)
    #Normalise data to get value between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    #Change dimension of numpy array mnist 784
    x_train = x_train.reshape([-1, 784])
    x_test = x_test.reshape([-1, 784])
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    return x_train, x_test, y_train, y_test

#One hot encoding of data ie convert classifical data to binary form
#Avoids giving higher numbers higher weightings
def one_hot(y):
    k = len(set(y))
    out = np.zeros((len(y), k))
    for i in range(len(y)):
        out[i, int(y[i])] = 1

    return out

#Fine tuning epochs number
def optimise_epochs():
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
def optimise_optimizer():
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam', 'Ftrl']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(x_train, y_train)
    # Summarise results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

#Fine tuning learning rate
def optimise_lrate():
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    param_grid = dict(learn_rate=learn_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(x_train, y_train)
    # Summarise results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

#Fine tuning activation function
def optimise_activation():
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    param_grid = dict(activation=activation)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(x_train, y_train)
    # Summarise results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()

    #Sequential model used since each layer has exactly one input and one output tensor
    model = Sequential([
        Dense(512, input_dim=784), #
        Activation('relu'), #Activation function
        Dense(512), #512 output neurons
        Activation('relu'), #Replaces negative values with zero and keeps positive values
        Dense(512),
        Activation('relu'),
        Dense(10),
        Activation('softmax') #Normalisation
    ])

    #Could change this optimiser 
    #Adam optimisation is a stochastic gradient descent method
    #Can handle sparse gradients on noisy problems
    optimiser = keras.optimizers.Adam(learning_rate=0.01)

    #Training configuration
    model.compile(
        loss='categorical_crossentropy', #Loss function for one hot inputs
        optimizer = optimiser,
        metrics=['accuracy'],
    )
    
  

    #Train the model
    print("Fit model on training data")
    model.fit(x_train, y_train, batch_size=64, epochs=15) #initially 15

    #Visualise model


    #Evaluate model on test data
    accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy}")

    #Generate predictions for 10 samples
    print("Predictions for 10 samples")
    predictions = model.predict(x_test[:10])
    print("Predictions shape: ", predictions.shape)
    
    y_new = model.predict_classes(x_test[:10])
   # display_data(x_test[:10], y_new)
    print("Visualising predictions")
    num_row = 2
    num_col = 5
    (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = mnist.load_data()
    images = x_test_orig[:10]
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col, 2*num_row))
    plt.title('Predictions')
    for i in range(len(images)):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(images[i], cmap='gray_r')
        ax.set_title('Label: {}'.format(y_new[i]))
   #     print("X=%s, Predicted=%s, Real=%s" % (x_test[:10][i], y_new[i]))        
    plt.show()

 #   optimise_epochs()





    