import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

from base import (
    read_dataset,
    shuffle,
    make_ground_truth_numeric,
    normalize,
    add_bias,
    extract_training_validation_testing_datasets,
    extract_X_y,
)


def random_weights(n: int, random_state: int):
    rand = np.random.RandomState(random_state)
    w = rand.normal(loc=0.0, scale=0.01, size=n)
    return w


def net_input(X, w):
    return np.dot(X, w)


def predict(X, y, w):
    y_pred = np.where(net_input(X, w) >= 0.0, 1, -1)
    num_correct_predictions = (y_pred == y).sum()
    accuracy = (num_correct_predictions / y.shape[0]) * 100
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return accuracy, tn, fp, fn, tp


def fit(X_train, y_train, X_validation, y_validation, eta=0.001, n_iter=10):
    m, n = X_train.shape
    v_m, _ = X_validation.shape
    training_error_iteration = np.zeros(n_iter)
    validation_error_iteration = np.zeros(n_iter)
    w = random_weights(n, random_state=1)
    for i in range(n_iter):
        error = 0
        for x, target in zip(X_train, y_train):
            pred = np.where(net_input(x, w) >= 0.0, 1, -1)
            delta = eta * (target - pred)
            w = w + delta * x
            error = error + int(delta != 0.0)
        training_error_iteration[i] = error
        error = 0
        for x, target in zip(X_validation, y_validation):
            pred = np.where(net_input(x, w) >= 0.0, 1, -1)
            delta = eta * (target - pred)
            error = error + int(delta != 0.0)
        validation_error_iteration[i] = error
    return w, training_error_iteration, validation_error_iteration


def plot_error(training_error_iteration, validation_error_iteration):
    plt.figure()
    plt.plot(training_error_iteration, label="training data")
    plt.plot(validation_error_iteration, label="validation data")
    plt.title("cost per iteration - Perceptron model")
    plt.ylabel("loss")
    plt.xlabel("No. epoch")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    path = "./dataset.csv"
    t = 0.7
    v = 0.1
    eta = 1e-4
    n_iter = 1000
    dataset = read_dataset(path)
    shuffle(dataset)
    nume_shuf_dataset = make_ground_truth_numeric(dataset)
    nor_nume_shuf_dataset = np.append(
        normalize(nume_shuf_dataset[:, :-1]),
        nume_shuf_dataset[:, -1].reshape(nume_shuf_dataset.shape[0], 1),
        axis=1,
    )
    temp = np.append(
        nor_nume_shuf_dataset[:, :-1],
        nor_nume_shuf_dataset[:, :-1] ** 2,
        axis=1,
    )
    second_order_nor_nume_shuf_dataset = np.append(
        temp,
        nor_nume_shuf_dataset[:, -1].reshape(nor_nume_shuf_dataset.shape[0], 1),
        axis=1,
    )
    b_second_order_nor_nume_shuf_dataset = add_bias(second_order_nor_nume_shuf_dataset)
    (
        training_ds,
        validation_ds,
        testing_ds,
    ) = extract_training_validation_testing_datasets(
        b_second_order_nor_nume_shuf_dataset, t=t, v=v
    )
    X_train, y_train = extract_X_y(training_ds)
    X_validation, y_validation = extract_X_y(validation_ds)
    X_test, y_test = extract_X_y(testing_ds)

    w, training_error_iteration, validation_error_iteration = fit(
        X_train, y_train, X_validation, y_validation, eta=eta, n_iter=n_iter
    )

    accuracy, tn, fp, fn, tp = predict(X_train, y_train, w)
    print("Perceptron accuracy on training data: %.2f%%" % accuracy)
    print("Perceptron tn, fp, fn, tp on training data:", tn, fp, fn, tp)
    accuracy, tn, fp, fn, tp = predict(X_test, y_test, w)
    print("Perceptron accuracy on testing data: %.2f%%" % accuracy)
    print("Perceptron tn, fp, fn, tp on testing data:", tn, fp, fn, tp)
    plot_error(training_error_iteration, validation_error_iteration)
