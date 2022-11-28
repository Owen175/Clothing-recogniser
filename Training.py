import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

class model():
  def __init__(self):
    self.fashion_mnist = keras.datasets.fashion_mnist  # load dataset

    (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.fashion_mnist.load_data()  # split into testing and training data

    self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



    self.train_images = self.train_images / 255.0

    self.test_images = self.test_images / 255.0

    #this makes it more compatible for the AI - the weights and biases will be closer to 0 or 1 which is what the thing scaled it down for


    #Passes through the layers sequentially
    eslf.model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1): This flattens the 2d array to a 1d array
        keras.layers.Dense(128, activation='relu'),  # hidden layer (2): all the neurones in previous layer connect to all in this layer - dense
        keras.layers.Dense(10, activation='softmax') # output layer (3)
    ])                                    #Softmax makes everything between 0 and one and makes them sum to one. activation functions. Outputs 10 different outcomes

    self.model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #sets the algorithms
    #This is called hyperperameter tuning - when you change the parameters

    model.fit(train_images, train_labels, epochs=1)  # we pass the data, labels and epochs
    #Fits the model to the training data. No input function because keras does it
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
    print('Test accuracy:', test_acc)
