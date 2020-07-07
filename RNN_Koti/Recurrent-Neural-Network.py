'''
# Recognising MNIST digits using a Recurrent Neural Network
# 
# Author: Koti Jaddu
'''

# Import of libraries
import tensorflow as tf
import tensorflow.keras.layers as KL
import numpy as np
import time

# Load the training and testing images along with their respective labels
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Keep the data between 0 and 1
train_images = (train_images/255)
test_images = (test_images/255)

# Define the shape of the training set
inputs = KL.Input(shape=(28, 28))

# Define the RNN model
x = KL.SimpleRNN(64, activation = "relu")(inputs)

# Define the output layer
outputs = KL.Dense(10, activation = "softmax")(x)

# Create the RNN model
model = tf.keras.models.Model(inputs, outputs)

# Compile and train the model using train_images
model.compile(
    optimizer =  "adam",
    loss = "sparse_categorical_crossentropy",
    metrics=["acc"] 
)
start_fit = time.time()
model.fit(train_images, train_labels, epochs = 20)               

# Calculate the accuracy from the predictions
start_eval = time.time()
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Print the accuracy and loss values for the user along with the time taken to train and make predictions
print(f"Loss: {test_loss}- Acc: {test_acc}- Time taken to train: {(time.time() - start_fit)/60} minutes- Time taken to predict: {time.time() - start_eval} seconds")