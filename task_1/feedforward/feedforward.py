import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers.core import *
from tensorflow.python.keras.optimizers import *
import tensorflow as tf
from keras.utils import plot_model
from keras import metrics
import sklearn.metrics as skl
import seaborn as sns
#Hyperparamter tuning can be imported from optimisation.py if necessary

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
    fig.canvas.set_window_title('Images Labelled from Training Set')
    for i in range(num):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(images[i], cmap='gray_r')
        ax.set_title('Label: {}'.format(labels[i]))
    fig.tight_layout()
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


#Methods for George
#display_data and one_hot methods also required (see top of file)

def fit(x_train, y_train, plot = True):
    #Data preprocessing
    if plot:
        display_data(x_train, y_train)
    x_train = x_train / 255.0
    x_train = x_train.reshape([-1, 784])
    y_train = one_hot(y_train)
    model = Sequential([
        Dense(512, input_dim=784),
        Activation('relu'),
        Dense(512),
        Activation('relu'),
        Dense(512),
        Activation('relu'),
        Dense(10),
        Activation('softmax')
    ])
    optimiser = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        loss='categorical_crossentropy', #Loss function for one hot inputs
        optimizer = optimiser,
        metrics=['accuracy'],
    )
    model.fit(x_train, y_train, validation_split=0.2, batch_size=64, epochs=20)
    return model

def predict(x_test, model, plot = True):
    #Normalise data to get value between 0 and 1
    x_test = x_test / 255.0
    #Change dimension of numpy array mnist 784
    x_test = x_test.reshape([-1, 784])

    y_pred = model.predict(x_test)
    smooth_predictions = []
    for row in y_pred:
        smooth_predictions.append(np.argmax(row))
    return smooth_predictions


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
    history = model.fit(x_train, y_train, validation_split=0.2, batch_size=64, epochs=20) #initially 15

    #Visualise model
    print("Model history keys")
    print(history.history.keys())

    #Plot model accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    #Plot model loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    #Evaluate model on test data
    accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy}")

    #Generate predictions for 10 samples
    print("Predictions for 10 samples")
    predictions = model.predict(x_test[:10])
    y_new = model.predict_classes(x_test[:10])
    y_pred = model.predict(x_test)
    print("Shape of predictions")
    print(y_pred.shape)

    #Generate confusion matrix

    # rounded_predictions = model.predict_classes(x_test, batch_size=128, verbose=0
    Y_pred = np.argmax(y_pred, 1)
    Y_test = np.argmax(y_test, 1)

    matrix = skl.confusion_matrix(Y_test, Y_pred)

    sns.heatmap(matrix.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix')
    plt.show()

    print("Visualising predictions")
    num_row = 2
    num_col = 5
    (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = mnist.load_data()
    images = x_test_orig[:10]
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col, 2*num_row))
    fig.canvas.set_window_title('Predicted Labels for 10 Samples')
    for i in range(len(images)):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(images[i], cmap='gray_r')
        ax.set_title('Label: {}'.format(y_new[i]))
    fig.tight_layout()
    plt.show()
