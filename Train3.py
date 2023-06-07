import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import Clean_Data as cd
from sklearn.neural_network import MLPClassifier

data_file = "traindata.txt"
labels_file = "trainlabels.txt"

X_train, X_val, y_train, y_val = cd.preprocess_data(data_file, labels_file)

best_accuracy = 0

for i in range(1, 100):
    for j in range(1, 100):

        MLPClass = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(i, j), random_state=0,
                                 solver='adam', max_iter=200, learning_rate_init=0.001, activation="relu")

        MLPClass.fit(X_train, y_train)

        y_pred = MLPClass.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)

        y_mat = confusion_matrix(y_val, y_pred)

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            print("Layer Size: ", i, j, "Accuracy:", accuracy, "Adam Relu")
            print(y_mat)
            print()


