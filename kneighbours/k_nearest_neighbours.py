'''
# Recognising MNIST digits using K Nearest Neighbours with parallel programming 
# 
# Author: Koti Jaddu
'''

# Import of libraries
import numpy as np 
import pandas as pd
import mnist
import time
import ray
import math

# THREADS_N = 16 # Constant holding the number of threads for parallel processes

# K_VALUE = 10 # K hyperparameter

def fit(x_train, y_train) :
    # Flatten each image from a 28x28 array to a 784 size vector
    x_train = x_train.reshape((-1, 784))
    return x_train, y_train


# Initialise parameters for parallel processing
ray.init()

# Routine that will be compatible with parallel processing which will predict the digit when given a 784 vector image
@ray.remote
def predict_digit(test_image, args, k_value):
    df = pd.DataFrame()
    x_train, y_train = args
    # Loop through the entire training images set
    for i in range(len(x_train)):

        # Calculate the Euclidean distance: sum of the errors for each index in the 784 vector test image with each training image
        euclidean_distance = np.sum(np.square(np.subtract(x_train[i], test_image, dtype=np.float64), dtype=np.float64))

        # Add this value along with the training label into a new row of the DataFrame df
        row = pd.Series([y_train[i], euclidean_distance])
        row_df = pd.DataFrame([row])
        df = pd.concat([row_df, df], ignore_index=True)

    # Rename the column names of the DataFrame df and sort the rows according to the euclidean_distance	column in ascending order
    df.columns = ['training_labels', 'euclidean_distance']
    df.sort_values(['euclidean_distance'], inplace=True)

    # Return the mode average of the training labels that are in the top k_value rows
    return int(df.head(k_value).mode()['training_labels'].iloc[0])



def predict(x_test, args, k_value = 10, threads_n = 8):
    output = []
    images_completed = 0

    # Flatten each image from a 28x28 array to a 784 size vector
    x_test = x_test.reshape((-1, 784))

    # Loop through each test image to be predicted and split the tasks into threads_n threads for parallel processing
    for i in range(len(x_test)//(threads_n)):
        result_ids = []

        # Create threads_n number of concurrent processes
        for j in range(threads_n):
            if threads_n * i + j < len(x_test):
                result_ids.append(predict_digit.remote(x_test[threads_n * i + j], args, k_value))
                images_completed += 1

        # Acquire the results
        results = ray.get(result_ids)

        # Add the results to what will be returned
        output.extend(results)

        # Print an update that shows the number of images predicted in batches of threads_n
        print(f"{images_completed} images predicted out of {len(x_test)}")

    return np.array(output)

# Load the training and testing images along with their respective labels
# train_images = mnist.train_images()
# train_labels = mnist.train_labels()
# test_images = mnist.test_images()
# test_labels = mnist.test_labels()

# Predict the first test image
# print(predict(test_images, fit(train_images, train_labels))[0])
# print(test_labels[0])