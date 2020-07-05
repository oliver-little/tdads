'''
# Recognising MNIST digits using K Nearest Neighbours without parallel programming 
# 
# Author: Koti Jaddu
'''

# Import of libraries
import numpy as np
import pandas as pd
import mnist
import time

K_VALUE = 10 # K hyperparameter

# Load the training and testing images along with their respective labels
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Flatten each image from a 28x28 array to a 784 size vector
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Routine that will predict the digit when given a 784 vector image
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

    # Rename the column names of the DataFrame df and sort the rows according to the euclidean_distance column in ascending order
    df.columns = ['target', 'euclidean_distance']
    df.sort_values(['euclidean_distance'], inplace=True)

    # Return the mode average of the training labels that are in the top K_VALUE rows
    return df.head(K_VALUE).mode()['target'].iloc[0]

# Initialise variables that keep track of the total correct predictions and the time it takes to predict them
total_correct = 0
start = time.time()

# Loop through each test image to be predicted
for i in range(len(test_images)):

    # Check the result to see if the prediction was correct and adjust the total_correct variable
    if predict_digit(test_images[i]) == test_labels[i]:
        total_correct += 1

    partial_end = time.time()

    # Print an update that shows the number of images predicted in batches of THREADS_N, the time it took, the accuracy so far and the estimated time to complete
    print(f"{i + 1} image(s) predicted in {partial_end-start} seconds - Accuracy: {total_correct / (i + 1)} - ETA: {((partial_end-start)/(3600 * (i + 1))) * (len(test_images) - (i + 1))} hours")

print()

# Print the final statement that includes the total number of images predicted, the time it took and the final accuracy
print(f"All {len(test_images)} images have been predicted in {(partial_end-start)/60} minutes with accuracy: {total_correct/len(test_images)}")