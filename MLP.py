# importing modules
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt
import Clean_Data

data_file = 'traindata.txt'
labels_file = 'trainlabels.txt'
x_train, x_test, y_train, y_test = Clean_Data.preprocess_data(data_file, labels_file)

# Define a list of model configurations
model_configs = [
    {'hidden_units': 256},
    {'hidden_units': 128},
    {'hidden_units': 64}
]

for config in model_configs:
    # Create a new model based on the configuration
    model = Sequential([
        # reshape 28 row * 28 column data to 28*28 rows
        # dense layer 1
        Dense(config['hidden_units'], activation='sigmoid'),
        # dense layer 2
        Dense(128, activation='sigmoid'),
        # output layer
        Dense(10, activation='sigmoid'),
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Evaluate the model
    results = model.evaluate(x_test, y_test, verbose=0)
    print('Model with', config['hidden_units'], 'hidden units - Test loss, Test accuracy:', results)
