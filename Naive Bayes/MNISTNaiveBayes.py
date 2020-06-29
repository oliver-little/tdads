## Implementation of Naive Bayes classifier on the MNIST dataset
## Resources used: https://lazyprogrammer.me/bayes-classifier-and-naive-bayes-tutorial-using/
##                 https://medium.com/data-sensitive/na%C3%AFve-bayes-tutorial-using-mnist-dataset-2b4a82b124d2

from scipy.stats import multivariate_normal as mvn
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

## HYPERPARAMETERS

# The constant to add to the variance to prevent division by zero errors
# Has an impact of at least 20% on prediction accuracy.
# Best accuracy around 1000-1100
VARIANCE_CONSTANT = 1025

class NaiveBayes():
    # Loads the MNIST dataset
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = tf.reshape(self.x_train, [len(self.x_train), 784])
        self.x_test = tf.reshape(self.x_test, [len(self.x_test), 784])

    # Trains the model by calculating prior probability, mean and variance for each class
    def train(self):
        separatedByClass = self.separateByClass(self.x_train, self.y_train)
        
        self.priorProbabilities = {}
        self.normalInputs = {}

        for c in range(10):
            self.normalInputs[c] = {}
            thisClassLength = len(separatedByClass[c])

            # Prior probability            
            self.priorProbabilities[c] = thisClassLength / len(self.x_train)

            # Mean and variance
            # Axis = 0 tells numpy to work by column in a 2D array, so each pixel is averaged rather than each image.
            self.normalInputs[c] = {
                "mean" : np.mean(separatedByClass[c], axis=0),
                "var" : np.var(separatedByClass[c], axis=0)
            }
            
            # Add constant to variance to prevent division by zero errors in the normal distribution
            # This does have an impact on prediction accuracy, so some tuning will need to be done here
            self.normalInputs[c]["var"] += VARIANCE_CONSTANT

    # Makes predictions for an array of images
    def predict(self, images):
        # Error handling
        if not hasattr(self, "normalInputs"):
            print("Train classifier first")
            return

        numImages, imageLength = images.shape
        numClasses = len(self.normalInputs)

        # Fill array of zeroes ready to insert probability calculations
        # Rows in this array represent an image
        # Columns represent a class
        # In any given row, the highest probability is the predicted class
        results = np.zeros((numImages, numClasses))
         
        for c, normalInput in self.normalInputs.items():
            # A bit to unpack here:
            # A multivariate normal distribution is used because calculating the probability for each pixel, for each image, for each class was far too slow.
            # Instead, what this does is calculate the probability for each pixel in all images in the prediction set at once
            # (the mean, variance and each image in the images array have a length of 784 so this works)
            # The results are returned as a logarithm to reduce the effect of floating point error with numbers close to 0 (If A > B then log(A) > log(B) so this is not an issue)
            
            # The results of the probability distribution are returned as an array of 10000 elements (one for each image),
            # and the log of the prior probability is added (rather than multiplied due to laws of logs)
            # results[:,c] accesses the c'th *column* of all rows in the array and writes to those elements.
            results[:,c] = mvn.logpdf(images, mean=normalInput["mean"], cov=normalInput["var"]) + np.log(self.priorProbabilities[c])

        # This selects the index of the highest probability for each image.
        # Axis = 1 tells numpy to work by row in a 2D array
        return np.argmax(results, axis=1)

    # Calculates the prediction accuracy of the classifier by predicting on the test set
    def score(self):
        results = self.predict(self.x_test)
        # Creates an array of True/False values, true if results[x] == self.y_test[x]
        correctPredictions = results == self.y_test
        
        return (sum(correctPredictions), len(correctPredictions))

    # Separates a dataset by the labels provided 
    def separateByClass(self, data, labels):
        if len(data) != len(labels):
            raise Exception("Length of data and labels arrays do not match.")
            
        byClass = {}
        for x in range(len(set(labels))):
            byClass[x] = []
            
        for x in range(len(data)):
            byClass[labels[x]].append(data[x])
            
        return byClass

    # Displays a heatmap of the mean values of the pixels for a given class
    def plotHeatmap(self, number):
        if not hasattr(self, "normalInputs"):
            print("Train classifier first")
            return
        else:
            plt.imshow(np.reshape(self.normalInputs[number]["mean"], (28, 28)), cmap="hot")
            plt.show()


if __name__ == "__main__":
    print("Loading dataset")
    startTime = time.time()
    classifier = NaiveBayes()
    endTime = time.time()
    print("Loading time: " + str(endTime - startTime) + "s")
    print("Starting Training")
    startTime = time.time()
    classifier.train()
    endTime = time.time()
    print("Training time: " + str(endTime - startTime) + "s")
    print("Starting Predictions")
    startTime = time.time()
    correct, total = classifier.score()
    endTime = time.time()
    print("Prediction time: " + str(endTime - startTime) + "s")
    print("Score: " + str(correct) + "/" + str(total))
    print("Percentage: " + str((correct/total)* 100) + "%")
    input()


