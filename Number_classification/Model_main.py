import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt
import pandas as pd


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print('X_train : ' + str(X_train.shape))
print('Y_train : ' + str(Y_train.shape))
print('X_test : ' + str(X_test.shape))
print('Y_test : ' + str(Y_test.shape))

#normalize 
X_train, X_test = X_train/255.0,X_test/255.0

model = tf.keras.models.Sequential(
    [               
            #specify input size
        ### START CODE HERE ### 
        tf.keras.layers.Flatten(input_shape=(28,28)), #this is to make the multi dimension input into 1 dimension input(flatten the dimension) to make it easier to process
        tf.keras.layers.Dense(128, activation='relu'), #dense like usual
        tf.keras.layers.Dropout(0.5), #dropout to reduce the overfitting (optional)
        tf.keras.layers.Dense(10), #10 because the output is divided to 10 (0,1,2,3,4,5,6,7,8,9) 10 output
        
        ### END CODE HERE ### 
    ], name = "my_model" 
)     
model.summary()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

model.fit(
    X_train, Y_train,
    epochs=5
)

model.evaluate(
    X_test, Y_test,
    verbose=True
)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
print(probability_model(X_test[:1]))