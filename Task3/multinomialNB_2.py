# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

np.set_printoptions(precision=6)


'''
    Function that load test and training set from CSV files.
    All columns with missing values are deleted.
    Function also returns train data set with calculate mean and std for every attribute.
'''
def load_dataset(filename_train, filename_test):

    # ------------------------ TRAIN SET ---------------------------
    dataTrain = pd.read_csv(filename_train, header=0)

    dataTrain = dataTrain.replace('?', np.nan)
    dataTrain = dataTrain.astype(float)

    #izbaci nedostajuca
    dataTrain = dataTrain.dropna()

    training_dict = {0.0 : [], 1.0 : []}

    df_zeros = dataTrain.loc[dataTrain['contraceptive'] == 0]
    df_ones = dataTrain.loc[dataTrain['contraceptive'] == 1]

    for row in  range(len(df_zeros.values.tolist())):
        training_dict[0.0].append(df_zeros.values.tolist()[row])

    for row in  range(len(df_ones.values.tolist())):
        training_dict[1.0].append(df_ones.values.tolist()[row])
        

    # ------------------------ TEST SET ---------------------------
    dataTest = pd.read_csv(filename_test, header=0)

    dataTest = dataTest.replace('?', np.nan)
    dataTest = dataTest.astype(float)

    #izbaci nedostajuce
    dataTest = dataTest.dropna()

    actual_values = dataTest['contraceptive'].tolist()
    test_set = dataTest.values.tolist()

    return training_dict, test_set, actual_values

def fit(training_dict, alpha = 1.0):

    #D is number of instances in training set
    D = len(training_dict[0.0]) + len(training_dict[1.0])
        
    class_log_prob = []
    for key in training_dict.keys():
        class_log_prob.append(np.log(1.0 * len(training_dict[key]) / D))
                                 
    #we need to count each values for each class and add self.alpha as smooting
    count = np.array([np.array(training_dict[key]).sum(axis=0) for key in training_dict]) + alpha

    feature_log_prob = np.log(count / count.sum(axis=1)[np.newaxis].T)

    return class_log_prob, feature_log_prob

def predict(X, class_log_prob, feature_log_prob):
    pred = [(feature_log_prob * x).sum(axis = 1) + class_log_prob for x in X]
    return np.argmax(pred, axis = 1)

