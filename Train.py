import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load the data from the text file into a NumPy array
input_data = np.loadtxt('traindata.txt', delimiter=',')

# Print the shape of the array (optional)
input_labels = np.loadtxt('trainlabels.txt', delimiter=',')

x_train = tf.constant(input_data[:int(len(input_data)*0.9)])
y_train = tf.constant(input_labels[:int(len(input_labels)*0.9)])
x_test = tf.constant(input_data[int(len(input_data)*0.9):])
y_test = tf.constant(input_labels[int(len(input_labels)*0.9):])

model = tf.keras.models.Sequential()

model.add(layers.Dense(71, activation="relu"))
model.add(layers.Dense(10))

model.

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    matrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=9000, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=1000, verbose=2)
