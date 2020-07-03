import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers.core import *
from tensorflow.python.keras.optimizers import *

'''A simple feed forward neural network with two hidden layers.'''

#Need to fine tune parameters: epochs, learning rate, 

mnist = tf.keras.datasets.mnist

class_names = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#Data preprocessing

def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #Normalise data to get value between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    #Change dimension of numpy arramnist 784
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

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()

    #Sequential model used since each layer has exactly one input and one output tensor
    model = Sequential([
        Dense(512, input_dim=784), 
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
    model.fit(x_train, y_train, batch_size=64, epochs=15)

    #Evaluate model on test data
    accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy}")

    #Generate predictions for 3 samples
    print("Predictions for 3 samples")
    predictions = model.predict(x_test[:3])
    print("Predictions shape: ", predictions.shape)








    