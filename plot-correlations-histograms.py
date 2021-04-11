import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from base import (
    read_dataset,
    shuffle,
    make_ground_truth_numeric,
    extract_training_validation_testing_datasets,
)


def plot_correlation_hitmap(corr, title):
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    path = "./dataset.csv"
    dataset = read_dataset(path)
    shuffle(dataset)
    nume_shuf_dataset = make_ground_truth_numeric(dataset)
    (
        training_ds,
        validation_ds,
        testing_ds,
    ) = extract_training_validation_testing_datasets(nume_shuf_dataset, t=0.7, v=0.1)
    sns.set_theme(style="white")
    training_corr = pd.DataFrame(training_ds).corr()
    bins = 100
    training_corr.plot.hist(bins=bins)
    plt.show()
    plot_correlation_hitmap(
        training_corr,
        "training dataset correlation hitmap (last column is ground truth)",
    )
    validation_corr = pd.DataFrame(validation_ds).corr()
    validation_corr.plot.hist(bins=bins)
    plt.show()
    plot_correlation_hitmap(
        validation_corr,
        "validation dataset correlation hitmap (last column is ground truth)",
    )
    testing_corr = pd.DataFrame(testing_ds).corr()
    testing_corr.plot.hist(bins=bins)
    plt.show()
    plot_correlation_hitmap(
        testing_corr, "testing dataset correlation hitmap (last column is ground truth)"
    )
