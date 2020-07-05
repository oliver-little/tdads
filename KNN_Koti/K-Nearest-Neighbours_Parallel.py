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


THREADS_N = 16 # Constant holding the number of threads for parallel processes

K_VALUE = 10 # K hyperparameter

# Load the training and testing images along with their respective labels
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Flatten each image from a 28x28 array to a 784 size vector
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Initialise parameters for parallel processing
ray.init()

# Routine that will be compatible with parallel processing which will predict the digit when given a 784 vector image
@ray.remote
def predict_digit(test_image):
    df = pd.DataFrame()

    # Loop through the entire training images set
    for i in range(len(train_images)):

        # Calculate the Euclidean distance: sum of the errors for each index in the 784 vector test image with each training images
        euclidean_distance = np.sum(np.square(np.subtract(train_images[i], test_image, dtype=np.float64), dtype=np.float64))

        # Add this value along with the training label into a new row of the DataFrame df
        row = pd.Series([train_labels[i], euclidean_distance])
        row_df = pd.DataFrame([row])
        df = pd.concat([row_df, df], ignore_index=True)

    # Rename the column names of the DataFrame df and sort the rows according to the euclidean_distance	column in ascending order
    df.columns = ['training_labels', 'euclidean_distance']
    df.sort_values(['euclidean_distance'], inplace=True)

    # Return the mode average of the training labels that are in the top K_VALUE rows
    return df.head(K_VALUE).mode()['training_labels'].iloc[0]

# Initialise variables that keep track of the total correct predictions and the time it takes to predict them
total_correct = 0
start = time.time()


# Loop through each test image to be predicted and split the tasks into THREADS_N threads for parallel processing
for i in range(len(test_images)//THREADS_N):
    result_ids = []

    # Create THREADS_N number of concurrent processes
    for j in range(THREADS_N):
        if THREADS_N * i + j < len(test_images):
            result_ids.append(predict_digit.remote(test_images[THREADS_N * i + j]))

    # Acquire the results
    results = ray.get(result_ids)

    # Check the results to see if the predictions were correct and adjust the total_correct variable
    for j in range(len(results)):
        if THREADS_N * i + j < len(test_images):
            if results[j] == test_labels[THREADS_N * i + j]:
                total_correct += 1
    images_completed = THREADS_N * (i + 1)

    partial_end = time.time()

    # Print an update that shows the number of images predicted in batches of THREADS_N, the time it took, the accuracy so far and the estimated time to complete
    print(f"{images_completed} image(s) predicted in {(partial_end-start)/60} minutes - Accuracy: {total_correct / images_completed} - ETA: {((partial_end-start)/(3600 * images_completed)) * (len(test_images) - images_completed)} hours")

print()

# Print the final statement that includes the total number of images predicted, the time it took and the final accuracy
print(f"All {len(test_images)} images have been predicted in {(partial_end-start)/3600} hours with accuracy: {total_correct/len(test_images)}")