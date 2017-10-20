import numpy as np
import pandas as pd
import math


def load_train_data(filename):
    """

    # This function load CSV data, and separate them into trainSet and label;

    :param filename: training data file name.
    :return: train_set: a matrix whose cols for attributes and rows for objects;
             label_set: a matrix for labels.


    """
    data = pd.read_csv(filename, sep=',')
    data_set = np.mat(data)
    [m,n] = data_set.shape
    a = np.ones((m, 1))

    train_set = data_set[:, 0: n-1]
    train_set = np.c_[train_set, a]

    label_set = data_set[:, n-1: n]

    return train_set, label_set


def load_test_data(filename):
    """

    This function load CSV Test data, and separate them into data and label
    :param filename: test data file name.
    :return: test_set: a matrix whose cols for attributes and rows for objects;
             test_label_set: a matrix for labels.


    """
    data = pd.read_csv(filename, sep=',')
    data_set = np.mat(data)
    [m, n] = data_set.shape
    a = np.ones((m,1))

    test_set = data_set[:, 0:n - 1]
    test_set = np.c_[test_set, a]

    test_label_set = data_set[:, n - 1: n]

    return test_set, test_label_set


def sigmoid(x):
    """

    This function compute the sigmoid output

    :param x: the inner product of weights and a specific attribute of all the objects
    :return: a float between 0 and 1


    """
    return 1.0 / (1 + math.exp(-x))


def logistic_regression(train_set, label_set, iter_num, learning_rate):
    """

    This function apply Gradient Ascend, to get the optimal weights.

    :param train_set: train set including all the attributes of all the objects, suppose the shape is m * n;
            m is the number of objects, n is the number of the attributes.
    :param label_set: label set including all the label of the objects, label is 0 or 1, suppose the shape is m * 1.
    :param iter_num: the times we train the model with the whole training set
    :param learning_rate: learning rate
    :return: weights: return the weights of all attributes


    """
    label_set = label_set.transpose()
    [m,n] = train_set.shape
    weights = np.ones((n, 1))
    for j in range(iter_num):
        for i in range(m):
            product = train_set[i] * weights
            h = sigmoid(product[0, 0])
            error = label_set[0, i] - h
            weights = weights + learning_rate * error * train_set[i].transpose()

    return weights


def fit(test_set, test_label_set, weights):
    """

    This function fit the test set into the trained model to do the prediction and compute the precision

    :param test_set: test data set with the formation being similar to the train set
    :param test_label_set: test label set with formation being similar to the label set
    :param weights: the weights for all attributes been computed in the function 'grad_ascend'
    :return: return the accuracy of the prediction


    """
    y_hat = test_set * weights
    y_hat_prob = []
    predict = []
    [m, n] = y_hat.shape
    for i in range(m):
        y_hat_prob.append(sigmoid(y_hat[i, 0]))
        if y_hat_prob[i] > 0.5:
            predict.append(1)
        else:
            predict.append(0)
    count = 0
    accuracy = 0
    for i in range(m):
        if predict[i] == test_label_set[i,0]:
            count = count + 1
        accuracy = float(count) / float(m)

    return accuracy
