# Preprocesses the dataset using PCA, and other methods
# PCA appears to increase the accuracy by ~1% at 0.95 

import tensorflow as tf
import numpy as np
import pickle
import time
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# The threshold at which a colour value is increased instead of reduced
CONTRAST_THRESHOLD = 80 # 80
CONTRAST_REDUCTION = 40 # 40
CONTRAST_INCREASE = 40 # 40

class MNISTSVMClassifier:
    def __init__(self, model=None):
        # Load data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        
        # Reshape the data from 28x28 to 784
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

        self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(self.x_train, self.y_train, test_size=0.25, random_state=1)
        
        if (model != None):
            self.clf = model
        else:
            self.clf = svm.SVC(kernel="rbf", C=1, gamma="scale")


    def train(self):
        # Currently using the first 10000 elements to keep training time relatively low
        self.clf.fit(self.x_train[:10000], self.y_train[:10000])

    def predict(self, images):
        return self.clf.predict(images)

    def score(self):
        results = self.clf.score(self.x_test, self.y_test)

        return (int(len(self.y_test) * results), len(self.y_test))

    def getClassifier(self):
        return self.clf

def increaseContrast(value):
    if value < CONTRAST_THRESHOLD:
        return min(0, value - CONTRAST_REDUCTION)
    else:
        return max(255, value + CONTRAST_INCREASE)

if __name__ == "__main__":
    print("Loading dataset")
    startTime = time.time()
    classifier = MNISTSVMClassifier()
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
    print("Saving trained model to MNISTSVMClassifier.pkl")
    with open("MNISTSVMClassifier.pkl", "wb") as file:
        pickle.dump(classifier.getClassifier(), file)
    input()
    
        
        
        
