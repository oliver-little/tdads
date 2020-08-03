import tensorflow as tf
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler


# read and transform data
# mnist = tf.keras.datasets.mnist
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
# X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
# Y_train = to_categorical(Y_train)
# Y_test = to_categorical(Y_test)
#
# use float32 and normalise between 0 and 1
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train = X_train / 255.0
# X_test = X_test / 255.0


# plot the digits
# plt.figure(figsize=(15,4.5))
# for i in range(30):
#     plt.subplot(3, 10, i+1)
#     plt.imshow(X_train[i].reshape((28,28)),cmap=plt.cm.binary)
#     plt.axis('off')
# plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
# plt.show()

num_nets = 15

def fit(x_train, y_train, plot = False, nets = 15, epochs = 45):
    global num_nets
    num_nets = nets

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    y_train = to_categorical(y_train)
    # Generate extra images! Keras does this for us.
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.10,
        width_shift_range = 0.1,
        height_shift_range = 0.1
    )


    # plot the newly-generated data
    if plot:
        X_train3 = X_train[9,].reshape((1,28,28,1))
        Y_train3 = Y_train[9,].reshape((1,10))
        plt.figure(figsize=(15,4.5))
        for i in range(30):
            plt.subplot(3, 10, i+1)
            X_train2, Y_train2 = datagen.flow(X_train3,Y_train3).next()
            plt.imshow(X_train2[0].reshape((28,28)),cmap=plt.cm.binary)
            plt.axis('off')
            if i==9: X_train3 = X_train[11,].reshape((1,28,28,1))
            if i==19: X_train3 = X_train[18,].reshape((1,28,28,1))
        plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
        plt.show()


    # Configure Neural Networks
    # This uses an adaptation of LeNet5's design
    models = [0] * nets
    for j in range(nets):
        models[j] = Sequential()
        models[j].add(Conv2D(32, kernel_size = 3, activation = 'relu', input_shape = (28, 28, 1)))
        models[j].add(BatchNormalization())
        models[j].add(Conv2D(32, kernel_size = 3, activation = 'relu'))
        models[j].add(BatchNormalization())
        models[j].add(Conv2D(32, kernel_size = 5, strides = 2, padding = 'same', activation = 'relu'))
        models[j].add(BatchNormalization())
        models[j].add(Dropout(0.4))

        models[j].add(Conv2D(64, kernel_size = 3, activation = 'relu'))
        models[j].add(BatchNormalization())
        models[j].add(Conv2D(64, kernel_size = 3, activation = 'relu'))
        models[j].add(BatchNormalization())
        models[j].add(Conv2D(64, kernel_size = 5, strides = 2, padding = 'same', activation = 'relu'))
        models[j].add(BatchNormalization())
        models[j].add(Dropout(0.4))

        models[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
        models[j].add(BatchNormalization())
        models[j].add(Flatten())
        models[j].add(Dropout(0.4))
        models[j].add(Dense(10, activation='softmax'))

        models[j].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


    # Decrease learning rate each epoch - converge on a better solution
    # (and don't accidentally jump way down)
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

    # do the training - takes ~2.5hrs for me (NVidia Compute Capability ~6.1)
    history = [0] * nets
    for j in range(nets):
      X_train2, X_val2, Y_train2, Y_val2 = train_test_split(x_train, y_train, test_size = 0.1)
      history[j] = models[j].fit_generator(datagen.flow(X_train2, Y_train2, batch_size=64),
                                           epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,
                                           validation_data = (X_val2, Y_val2), callbacks = [annealer])
      print("CNN {0:d}: Epochs={1:d}, Train Acc.={2:.5f}, Validation Acc.={3:.5f}".format(j+1, epochs, max(history[j].history['acc']),max(history[j].history['val_acc'])))
    return history, models

def predict(x_test, args, plot=False):
    history, models = args
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # predict for test data
    results = np.zeros((x_test.shape[0], 10))
    for j in range(num_nets):
      results = results + models[j].predict(x_test)
    results = np.argmax(results, axis=1)
    results = pd.Series(results, name='Label')

    # plot the results
    if plot:
        plt.figure(figsize=(15,6))
        for i in range(40):
            plt.subplot(4, 10, i+1)
            plt.imshow(X_test[i].reshape((28,28)),cmap=plt.cm.binary)
            plt.title("predict=%d" % results[i],y=0.9)
            plt.axis('off')
        plt.subplots_adjust(wspace=0.3, hspace=-0.1)
        plt.show()

    return results
