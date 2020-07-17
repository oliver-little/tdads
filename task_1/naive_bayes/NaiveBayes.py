## Implementation of Naive Bayes classifier on the MNIST dataset
## Resources used: https://lazyprogrammer.me/bayes-classifier-and-naive-bayes-tutorial-using/
##                 https://medium.com/data-sensitive/na%C3%AFve-bayes-tutorial-using-mnist-dataset-2b4a82b124d2

from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import numpy as np

## HYPERPARAMETERS

# The constant to add to the variance to prevent division by zero errors
# Has an impact of at least 20% on prediction accuracy.
# Best accuracy around 1000-1100
VARIANCE_CONSTANT = 1025

class NaiveBayes():
    # Trains the model by calculating prior probability, mean and variance for each class
    def train(self, x_train, y_train):
        separatedByClass = self.separateByClass(x_train, y_train)
        
        self.priorProbabilities = {}
        self.normalInputs = {}

        for c in range(10):
            self.normalInputs[c] = {}
            thisClassLength = len(separatedByClass[c])

            # Prior probability            
            self.priorProbabilities[c] = thisClassLength / len(x_train)

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
    # Images should be 784 long, and the array should be a numpy array
    def predict(self, x_test):
        # Error handling
        if not hasattr(self, "normalInputs"):
            print("Train classifier first")
            return

        numImages, imageLength = x_test.shape
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
            results[:,c] = mvn.logpdf(x_test, mean=normalInput["mean"], cov=normalInput["var"]) + np.log(self.priorProbabilities[c])

        # This selects the index of the highest probability for each image.
        # Axis = 1 tells numpy to work by row in a 2D array
        return np.argmax(results, axis=1)   

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

    def generateConfusionMatrix(self, x_test, y_test):
        # Adapted from: https://stackoverflow.com/questions/50021928/build-confusion-matrix-from-two-vector
        results = self.predict(x_test)

        confusionMatrix = np.zeros((10, 10), dtype=int)
        np.add.at(confusionMatrix, (results, y_test), 1)
        return confusionMatrix

    def displayConfusionMatrix(self, x_test, y_test):
        matrix = self.generateConfusionMatrix(x_test, y_test)

        print("Confusion matrix: ")
        print()
        print(" " + "".join('%6d' % num for num in range(10)))
        for y in range(len(matrix)):
            row = matrix[y]
            print(str(y), end="")
            for x in row:
                print("{:6}".format(x), end="")
            print()
            
    # Displays a heatmap of the mean values of the pixels for a given class
    def plotHeatmap(self, number):
        if not hasattr(self, "normalInputs"):
            print("Train classifier first")
            return
        else:
            plt.imshow(np.reshape(self.normalInputs[number]["mean"], (28, 28)), cmap="hot")
            plt.show()

def fit(x_train, y_train):
    clf = NaiveBayes()
    x_train = np.asarray(x_train).reshape((len(x_train), 784))
    clf.train(x_train, y_train)
    return [clf]

def predict(x_test, fit_return_list):
    clf = fit_return_list[0]
    x_test = np.asarray(x_test).reshape((len(x_test), 784))
    return clf.predict(x_test)

