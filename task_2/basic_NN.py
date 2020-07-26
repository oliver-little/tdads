# -*- coding: utf-8 -*-
"""
Imports data created by TweetDict,trains a net on most of it, and tests on the rest.
The input vectors represent words I've established as predicting positive or negative
sentiment, based on the given data. 

The size of this vector depends on the settings in TweetDict, but are derived 
from the files, so are not critical. Typically 1500 - 2000 dimensions.

The labels are the values given on the given csv file: they get converted to 
0 for neutral
1 for positive
2 for negative

Both are loaded into numpy arrays, and passed to a TensorFlow model in keras.
The first 10,000 entries are used for training, and the rest (~4640) for testing. 

Created on Wed Jun 24 10:29:06 2020
@author: David Marples
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

test_vectors = np.load("TweetVectors.npy")

f = open("true_sentiment.txt","r")
labels = []
for x in f:
    x = x[:-1]
    if x == 'positive':
        labels.append(1)
    elif x == 'negative':
        labels.append(2)
    elif x == 'neutral':
        labels.append(0)
f.close()


test_labels = np.array(labels)

model = keras.Sequential([
    keras.layers.Input(len(test_vectors[0])),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(3)
    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(test_vectors[:10000], test_labels[:10000], epochs=10)

test_loss, test_acc = model.evaluate(test_vectors[10000:], test_labels[10000:], verbose=2)

print('\nTest accuracy:', test_acc)

