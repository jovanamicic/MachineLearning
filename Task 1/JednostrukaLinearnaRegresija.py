#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:29:31 2017

@author: student
"""

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt


# Linear Regression With Stochastic Gradient Descent for Wine Quality
from random import seed
from random import randrange
from csv import reader
from math import sqrt

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# Evaluate an algorithm using a cross validation split
#   do predict mozemo sve obrsati
def evaluate_algorithm(dataset, *args):
    lenght = int(len(dataset) * 0.8)
    train_set = dataset[:lenght]
    test_set = dataset[lenght:]
    predicted = linear_regression_sgd(train_set, test_set, *args)
    actual = [row[-1] for row in test_set]
    rmse = rmse_metric(actual, predicted)
    return rmse

# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return yhat

# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
            # print(l_rate, n_epoch, error)
    return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
def linear_regression_sgd(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coef)
        predictions.append(yhat)
    return(predictions)

# Linear Regression on wine quality dataset
# load and prepare data
filename = 'ships.csv'
dataset = load_csv(filename)

header = ['X','Y']
data = pd.read_csv(filename, names=header, header = 1)

x = data['X']
y = data['Y']

dataset = dataset[1:]
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# normalize
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
l_rate = 0.01
n_epoch = 50
score = evaluate_algorithm(dataset, l_rate, n_epoch)

plt.plot(x, y, 'o')

print('Mean RMSE: %.3f' % score)

#def main(argv):
#    #Read file
#    fileName = argv[1]
#    header = ['X','Y']
#    data = pd.read_csv(fileName, names=header, header = 1)
#
#    x = data['X']
#    y = data['Y']
#    np.plot(x,y,'o')
#
#    dataset = []
#    for xi,yi in zip(x,y):
#        dataset.append([xi,yi])
#        
#    p2 = np.polyfit(x,y,2)
#    np.plot(x, np.polyval(p2,x), 'r-')
#      
#if __name__ == "__main__":
#    main(sys.argv) 
