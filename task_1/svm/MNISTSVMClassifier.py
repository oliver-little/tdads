## Basic SVM Classifier using Sklearn 

import tensorflow as tf
import pickle
import time
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

# Hyperparameters
#
# Preprocessing Scaler:
#   - MinMaxScaler: scales the dataset to values between 0 and 1 (seems to give by far the best accuracy)
#   - StandardScaler: scales the dataset to have a mean of zero and unit variance (gives poor accuracy, likely because it doesn't preserve the zero value of large portions of the images)
#
# Kernel trick:
#   - poly
#       - degree: changes the degree of the polynomial kernel
#   - rbf
#       - gamma: adjusts the gamma parameter of the rbf (higher = more overfitting)
#
# C: adjusts the C parameter for the SVM (higher = more overfitting)


class MNISTSVMClassifier:
    def __init__(self, model=None):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = tf.reshape(self.x_train, [len(self.x_train), 784])
        self.x_test = tf.reshape(self.x_test, [len(self.x_test), 784])
        scaler = MinMaxScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        if (model != None):
            self.clf = model
        else:
            self.clf = svm.SVC(kernel="rbf", C=1, gamma="scale") # These parameters are also the defaults for the SVM classifier


    def train(self):
        self.clf.fit(self.x_train[:10000], self.y_train[:10000])

    def predict(self, images):
        return self.clf.predict(images)

    def score(self):
        results = self.clf.score(self.x_test, self.y_test)

        return (len(self.y_test) * results, len(self.y_test))

    def getClassifier(self):
        return self.clf


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
    
        
        
        
