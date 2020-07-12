# Preprocesses the dataset using PCA, and other methods
# PCA appears to increase the accuracy by ~1% at 0.95 

import tensorflow as tf
import numpy as np
import time
import csv
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.stats import uniform

# The subset of the data to work on
FIT_SUBSET = 20000

# The threshold at which a colour value is increased instead of reduced
CONTRAST_THRESHOLD = 70 # 70
CONTRAST_REDUCTION = 80 # 80
CONTRAST_INCREASE = 70 # 70

class MNISTSVMClassifier:
    def __init__(self):
        # Load data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        
        # Reshape the data from 28x28 to 784 and apply contrast increase
        self.x_train = tf.reshape(self.x_train, [len(self.x_train), 784])
        vectorisedContrast = np.vectorize(increaseContrast)
        self.x_train = vectorisedContrast(self.x_train.numpy())
        self.x_test = tf.reshape(self.x_test, [len(self.x_test), 784])
        self.x_test = vectorisedContrast(self.x_test.numpy())

        # Set up PCA then fit to the training data.
        # The parameter represents the variance to retain (in this case retain 90% of the original variance)
        scaler = PCA(0.9).fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

        # Work on a subset of the data to increase training speed
        self.x_train = self.x_train[:FIT_SUBSET]
        self.y_train = self.y_train[:FIT_SUBSET]
        
        self.estimator = svm.SVC()
        params = [
            {
                "kernel" : ["rbf"],
                "C" : [1, 10, 100, 1000],
                "gamma" : ["scale", 1e-7, 2e-7, 2e-8]
            },
            {
                "kernel" : ["poly"],
                "C": [1, 10, 100, 1000],
                "degree" : [3]
            }
        ]
        self.clf = GridSearchCV(self.estimator, param_grid=params, n_jobs=5, verbose=3)

    # Finds the optimal hyperparameters using randomised search
    def train(self):
        self.clf.fit(self.x_train, self.y_train)
        
        print("Best parameters set found on development set:")
        print()
        print(self.clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = self.clf.cv_results_['mean_test_score']
        stds = self.clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, self.clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()
    
        print("Outputting to csv")
        # Save data as CSV
        with open("cv_results_rbf.csv", "w") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["Iteration", "Mean" ,"Std", "Params"])
            for row in range(len(means)):
                writer.writerow([row, means[row], stds[row], self.clf.cv_results_["params"][row]])
                

    def score(self):
        results = self.clf.score(self.x_test, self.y_test)

        return (int(len(self.y_test) * results), len(self.y_test))

def increaseContrast(value):
    if value < CONTRAST_THRESHOLD:
        return min(0, value - CONTRAST_REDUCTION)
    else:
        return max(255, value + CONTRAST_INCREASE)

if __name__ == "__main__":
    print("Loading and preprocessing dataset")
    startTime = time.time()
    classifier = MNISTSVMClassifier()
    endTime = time.time()
    print("Loading and preprocessing time: " + str(endTime - startTime) + "s")
    print("Starting Randomised Search")
    startTime = time.time()
    classifier.train()
    endTime = time.time()
    print("Search time: " + str(endTime - startTime) + "s")
    input()
    
        
        
        
