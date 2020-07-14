'''
# Recognising MNIST digits using a Recurrent Neural Network
# 
# Author: Koti Jaddu
'''

# Import of libraries
import tensorflow as tf
import tensorflow.keras.layers as KL
import numpy as np

def fit(x_train, y_train, epochs = 20):
    # Keep the data between 0 and 1
    x_train = (x_train/255)

    # Define the shape of the training set
    inputs = KL.Input(shape=(28, 28))

    # Define the RNN model
    x = KL.SimpleRNN(64, activation = "relu")(inputs)

    # Define the output layer
    outputs = KL.Dense(10, activation = "softmax")(x)

    # Create the RNN model
    model = tf.keras.models.Model(inputs, outputs)

    # Compile and train the model using x_train
    model.compile(
        optimizer =  "adam",
        loss = "sparse_categorical_crossentropy",
        metrics=["acc"] 
    )
    model.fit(x_train, y_train, epochs = epochs)

    return model

def predict(x_test, model):
    # Keep the data between 0 and 1
    x_test = (x_test/255)

    # Return the predictions for the test images as a list
    return np.argmax(model.predict(x_test), axis = 1)

# Load the training and testing images along with their respective labels
# mnist = tf.keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Predict the first test image
# print(predict(test_images, fit(train_images, train_labels, epochs = 2))[0])
# print(test_labels[0])