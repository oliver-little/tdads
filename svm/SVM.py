# Preprocesses the dataset using PCA, and increasing image contrast
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA

# The threshold at which a colour value is increased instead of reduced
# These values were calculated using manual testing
CONTRAST_THRESHOLD = 70 # 70
CONTRAST_REDUCTION = 80 # 80
CONTRAST_INCREASE = 70 # 70

def increaseContrast(value):
    if value < CONTRAST_THRESHOLD:
        return min(0, value - CONTRAST_REDUCTION)
    else:
        return max(255, value + CONTRAST_INCREASE)

VECTORISED_CONTRAST = np.vectorize(increaseContrast)

def fit(x_train, y_train):
    clf = svm.SVC(kernel="rbf", C=10, gamma=2e-7)
    x_train = VECTORISED_CONTRAST(np.asarray(x_train).reshape((len(x_train), 784)))
    scaler = PCA(0.9).fit(x_train)
    x_train = scaler.transform(x_train)
    clf.fit(x_train, y_train)
    return [clf, scaler]

def predict(x_test, fit_return_list):
    clf = fit_return_list[0]
    x_test = VECTORISED_CONTRAST(np.asarray(x_test).reshape((len(x_test), 784)))
    x_test = fit_return_list[1].transform(x_test)
    return clf.predict(x_test)
        
        
        
