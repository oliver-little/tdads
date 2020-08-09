import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import keras.utils

class LogisticRegressor():

    def __init__(self, input, output):

        self.input_size = input
        self.output_size = output

    def load_model(self):

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.output_size, input_dim=self.input_size, activation='softmax'))

    def run_model(self, train_images, train_labels):

        train_labels = np_utils.to_categorical(train_labels, self.output_size)

        epochs = 30
        batch_size = 128


        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(train_images, train_labels ,
                            batch_size=batch_size, epochs=epochs,
                            verbose=1)

    def predict_model(self, test_images):

        predictions = self.model.predict(test_images)
        return predictions.argmax(axis=1)

def fit(x_train, y_train, input, output):
    regressor = LogisticRegressor(input, output)
    regressor.load_model()
    regressor.run_model(train_images=x_train, train_labels=y_train)

    return [regressor]

def predict(images, reg):

    regressor = reg[0]
    return regressor.predict_model(images)

def evaluate(predictions, labels):

    index = 0
    correct = 0

    for i in predictions:
        if i == labels[index]:
            correct += 1
        index += 1

    return correct / index
"""
UNCOMMENT TO RUN WITHIN FILE

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

reg = fit(train_images, train_labels, 784, 10)
predictions = predict(test_images, reg)
print(predictions)

for x in range(25):
    print(predictions[x], "  |  ", test_labels[x])

print(predictions.shape)
print(test_labels)
print(test_labels.min(), predictions.min())
print(test_labels.max(), predictions.max())

print(evaluate(predictions, test_labels))
"""
