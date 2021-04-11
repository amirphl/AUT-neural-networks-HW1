import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing


def read_dataset(path):
    return pd.read_csv(path).values


def shuffle(dataset):
    np.random.shuffle(dataset)


def make_ground_truth_numeric(dataset):
    labels = np.unique(dataset[:, -1])
    numbers = list(range(-1, len(labels)))
    numbers.remove(0)
    for label, number in zip(labels, numbers):
        dataset[dataset[:, -1] == label, -1] = number
    return dataset.astype(float)


def normalize(dataset):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(dataset)


def add_bias(dataset):
    m, _ = dataset.shape
    bias = np.ones(m)
    return np.insert(dataset, 0, values=bias, axis=1)


def extract_training_validation_testing_datasets(dataset, t, v):
    assert t + v < 1
    m, n = dataset.shape
    n = n - 1
    m_train = int(t * m)
    m_validation = int(v * m)
    m_tv = m_train + m_validation
    training_ds = dataset[:m_train]
    validation_ds = dataset[m_train:m_tv]
    testing_ds = dataset[m_tv:]
    return training_ds, validation_ds, testing_ds


def extract_X_y(dataset):
    return dataset[:, :-1], dataset[:, -1]
