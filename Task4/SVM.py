# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.svm import SVC

np.set_printoptions(precision=6)


'''
    Function that load test and training set from CSV files.
    All columns with missing values are deleted.
    Function return training dataset separated by classes, and it returns
    test features and test actual values.
'''
def load_dataset(filename_train, filename_test):

    # ------------------------ TRAIN SET ---------------------------
    dataTrain = pd.read_csv(filename_train, header=0)

    dataTrain = dataTrain.replace('?', np.nan)
    dataTrain = dataTrain.astype(float)

    # izbaci nedostajuca
    dataTrain = dataTrain.dropna()

    # zameni nedostajuce sa mean te kolone
    # dataTrain = dataTrain.apply(lambda x: x.fillna(x.mean()), axis=0)
    
    training_set = dataTrain.values.tolist()


    # ------------------------ TEST SET ---------------------------
    dataTest = pd.read_csv(filename_test, header=0)

    dataTest = dataTest.replace('?', np.nan)
    dataTest = dataTest.astype(float)

    # izbaci nedostajuce
    dataTest = dataTest.dropna()

    # zameni nedostajuce sa mean te kolone
    # dataTest = dataTest.apply(lambda x: x.fillna(x.mean()), axis=0)

    actual_values = dataTest['class'].tolist()
    test_set = dataTest.drop('class', 1).values.tolist()
    return training_set, test_set, actual_values