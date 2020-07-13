import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.linear_model import LinearRegression
import math
from sklearn import metrics

class LinearRegressor:

  def __init__(self):
    self.input_size = 784

  def run_model(self, train_images, train_labels):
    
    #reshape and normalize the data
    train_images = train_images.reshape(60000, self.input_size)  
    train_images = train_images.astype('float32')      
    train_images /= 255
    self.model = LinearRegression()
    self.model.fit(train_images, train_labels)



  def predict_model(self, test_images):
    
      #reshape and normalize the data
      test_images = test_images.reshape(10000, self.input_size)
      test_images = test_images.astype('float32')
      test_images /= 255
        
      predictions = self.model.predict(test_images) 

      return predictions



  def evaluate_model(self, test_images, test_labels):
    r_sq = model.score(test_images, test_labels)

    predictions = model.predict(test_images)

    i = 0
    score = 0
    for x in predictions:
      if (int(x) == self.test_labels[i]):
          score += 1
      i += 1

    accuracy = score / 10000

    return r_sq, predictions, accuracy


def fit(x_train, y_train):
    regressor = LinearRegressor()
    regressor.run_model(train_images=x_train, train_labels=y_train)

    return [regressor]

def predict(images, reg):

    regressor = reg[0]
    return regressor.predict_model(images)
    
"""
UNCOMMENT TO RUN WITHIN FILE

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
reg = fit(train_images, train_labels)
predictions = predict(test_images, reg)
print(predictions)

"""
