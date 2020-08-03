import tensorflow as tf
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

# Trains a number of decision trees on random samples of the dataset, and makes predictions by majority rule of all estimators.
# With enough estimators, accuracy can be very high.

# Hyperparameters
#
# num_estimators: number of decision trees to use for predicting.
#   Value of ~200 is a good value from manual testing.
# max_features: number of features of the input set to consider when looking for the best feature to split on - considering more features hugely increases training time but also doesn't seem to increase accuracy for this dataset.
#   Default is auto/sqrt, sqrt of num_features, but other options are log2 or None for all features

# The preprocessing used in the SVM classifier actually decreased the accuracy of this classifier.
# Both PCA and increasing the contrast had a negative impact.

class MNISTRandomForestClassifier:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = tf.reshape(self.x_train, [len(self.x_train), 784])
        self.x_test = tf.reshape(self.x_test, [len(self.x_test), 784])
        
        self.clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=1)

    def train(self):
        self.clf.fit(self.x_train, self.y_train)

    def predict(self, images):
        return self.clf.predict(images)

    def score(self):
        results = self.predict(self.x_test)

        correctPredictions = results == self.y_test

        return (sum(correctPredictions), len(correctPredictions))

    def getClassifier(self):
        return self.clf

if __name__ == "__main__":
    print("Loading dataset")
    startTime = time.time()
    classifier = MNISTRandomForestClassifier()
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
    print("Percentage: " + str((correct/total)*100))
    print("Saving trained model to MNISTRandomForestClassifier.pkl")
    input()
    
        
        
        
