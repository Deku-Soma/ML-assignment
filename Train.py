import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import Clean_Data as cd


data_file = 'traindata.txt'
labels_file = 'trainlabels.txt'
X_train, X_val, y_train, y_val = cd.preprocess_data(data_file, labels_file)

x_train = tf.constant(X_train)
y_train = tf.constant(y_train)
x_test = tf.constant(X_val)
y_test = tf.constant(y_val)

model = tf.keras.models.Sequential()

model.add(layers.Dense(71, activation="relu"))
model.add(layers.Dense(10))

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    matrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=9000, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=1000, verbose=2)
