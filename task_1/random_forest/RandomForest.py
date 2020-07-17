import numpy as np
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

def fit(x_train, y_train):
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    x_train = np.asarray(x_train).reshape((len(x_train), 784))
    clf.fit(x_train, y_train)
    return [clf]

def predict(x_test, fit_return_list):
    clf = fit_return_list[0]
    x_test = np.asarray(x_test).reshape((len(x_test), 784))
    return clf.predict(x_test)

        
        
        
