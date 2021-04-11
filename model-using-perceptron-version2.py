import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from base import (
    read_dataset,
    shuffle,
    make_ground_truth_numeric,
    add_bias,
    extract_training_validation_testing_datasets,
    extract_X_y,
)

if __name__ == "__main__":
    path = "./dataset.csv"
    t = 0.7
    v = 0.1
    epochs = 10000
    batch_size = 100
    dataset = read_dataset(path)
    shuffle(dataset)
    nume_shuf_dataset = make_ground_truth_numeric(dataset)
    b_nume_shuf_dataset = add_bias(nume_shuf_dataset)
    (
        training_ds,
        validation_ds,
        testing_ds,
    ) = extract_training_validation_testing_datasets(b_nume_shuf_dataset, t=t, v=v)
    X_train, y_train = extract_X_y(training_ds)
    X_validation, y_validation = extract_X_y(validation_ds)
    X_test, y_test = extract_X_y(testing_ds)
    _, n = X_train.shape

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(
            1,
            input_shape=(n,),
            activation=tf.keras.activations.hard_sigmoid,
            kernel_initializer="glorot_uniform",
        )
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_validation, y_validation),
    )
    _, accuracy = model.evaluate(X_test, y_test)
    print("accuracy on testing dataset: %0.3f" % accuracy)
    y_pred = model.predict(X_test)
    y_pred[y_pred[:, 0] >= 0.5, :] = 1  # hard sigmoidal function
    y_pred[y_pred[:, 0] < 0.5, :] = 0  # hard sigmoidal function
    # unique, counts = np.unique(y_pred, return_counts=True)
    # print(dict(zip(unique, counts)))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("tn, fp, fn, tp:", tn, fp, fn, tp)
    plt.plot(history.history["loss"], label="training data")
    plt.plot(history.history["val_loss"], label="validation data")
    plt.title("cost per iteration - perceptron model")
    plt.ylabel("loss")
    plt.xlabel("No. epoch")
    plt.legend()
    plt.show()
