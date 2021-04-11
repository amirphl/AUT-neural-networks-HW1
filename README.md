
# binary classification using single Perceptron unit, single Adaline unit, and second-order Perceptron

The project implements a mineral dataset binary classification using single-layer feed-forward networks (SLFFN) by utilizing Perceptron, second-order Perceptron, and Adaline units in the hidden layer in this homework.
the training cost history and validation cost history are available in figures [here](https://github.com/amirphl/AUT-neural-networks-HW1/tree/main/figures),
I strongly recommend reading [this article](https://pabloinsente.github.io/the-adaline) to learn the mathematical background, Perceptron and Adaline units, algorithms used to train, differences, and...


# dataset
- features: 60 (without bias column)
- ground truth labels: M, R
- column names: unknown

## techniques used to preprocess the data
- shuffling
- normalization using the min-max scaler
- adding bias
- adding polynomial features (for second-order Perceptron)
- splitting dataset to training data (70%), validation data (10%), and testing data (20%)

## hyper parameters
- eta: fixed (1e-4) (This is also called the learning rate.)
- iterations (epochs for Adaline SLFFN): tuned with the validation data

## evaluation metrics
- accuracy, confusion matrix(TP, TN, FP, FN)

## problems and solutions
- Explain the architecture of Perceptron and Adaline units and differences. solution: [Problem 1 section](https://github.com/amirphl/AUT-neural-networks-HW1/blob/main/neural_networks_HW1_AUT_99131006_amir_pirhosseinloo.pdf)
- Plot the dataset feature correlations histogram and its hit map. Which features are candidates to be removed? solution: [Problem 2 section](https://github.com/amirphl/AUT-neural-networks-HW1/blob/main/neural_networks_HW1_AUT_99131006_amir_pirhosseinloo.pdf)
- Implement Perceptron unit and binary classification. Plot training and validation errors per iteration. At which iteration the network converges? solution: [Problem 3 section](https://github.com/amirphl/AUT-neural-networks-HW1/blob/main/neural_networks_HW1_AUT_99131006_amir_pirhosseinloo.pdf)
- Repeat the previous problem for the Adaline unit then compare it by Perceptron in terms of accuracy, confusion matrix, and convergence speed. solution: [Problem 4 section](https://github.com/amirphl/AUT-neural-networks-HW1/blob/main/neural_networks_HW1_AUT_99131006_amir_pirhosseinloo.pdf)
- Repeat the previous problem with second-order Perceptron. solution: [Problem 5 section](https://github.com/amirphl/AUT-neural-networks-HW1/blob/main/neural_networks_HW1_AUT_99131006_amir_pirhosseinloo.pdf)

## conclusion
Generally, second-order Perceptron and Adaline units perform better than a Perceptron unit in terms of accuracy. Also, the two mentioned models converge faster than Perceptron.

![training-data-correlation-hitmap](https://github.com/amirphl/AUT-neural-networks-HW1/blob/main/figures/training-data-correlation-hitmap.png?raw=true)
![perceptron-cost-history](https://github.com/amirphl/AUT-neural-networks-HW1/blob/main/figures/perceptron-cost-history.png?raw=true)
![adaline-cost-history](https://github.com/amirphl/AUT-neural-networks-HW1/blob/main/figures/adaline-cost-history.png?raw=true)
![second-order-perceptron-cost-history](https://github.com/amirphl/AUT-neural-networks-HW1/blob/main/figures/second-order-perceptron-cost-history.png?raw=true)

## references
- [https://pabloinsente.github.io/the-adaline](https://pabloinsente.github.io/the-adaline)

