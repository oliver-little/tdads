# Preprocesses the dataset using PCA, and increasing image contrast


import tensorflow as tf
import numpy as np
import time
from sklearn import svm
from sklearn.decomposition import PCA

# The threshold at which a colour value is increased instead of reduced
# These values were calculated using manual testing
CONTRAST_THRESHOLD = 70 # 70
CONTRAST_REDUCTION = 80 # 80
CONTRAST_INCREASE = 70 # 70

class MNISTSVMClassifier:
    def __init__(self, model=None):
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
        
        if (model != None):
            self.clf = model
        else:
            # Hyperparameters from randomised search
            self.clf = svm.SVC(kernel="rbf", C=10, gamma=2e-7)


    def train(self):
        self.clf.fit(self.x_train, self.y_train)

    def predict(self, images):
        return self.clf.predict(images)


    # Score on test set
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
    print("Loading and preprocessing dataset")
    startTime = time.time()
    classifier = MNISTSVMClassifier()
    endTime = time.time()
    print("Loading and preprocessing time: " + str(endTime - startTime) + "s")
    
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
    input()
    
        
        
        
