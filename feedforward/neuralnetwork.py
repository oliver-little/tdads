import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

# From https://www.tensorflow.org/tutorials/quickstart/beginner

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#Display first 25 digits from the training set with their labels

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()

model = tf.keras.models.Sequential([
  #Transforms images from a 2-D array to a 1-D array (no parameters to learn)
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  #Neural layer with 128 nodes
  tf.keras.layers.Dense(128, activation='relu'),
  #Returns a logits array of length 10 (for the number of digits 0-9)
  #Each node has a score that the current image belongs to one of the digits
  tf.keras.layers.Dense(10)
])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


model.compile(
              #How model is updated based on data and loss function
              optimizer='adam',
              #Accuracy of model during training
              loss=loss_fn,
              #Monitor training and testing based on fraction of images correctly classified 
              metrics=['accuracy'])

#Train the model - fit the model to the training data
model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(x_test)