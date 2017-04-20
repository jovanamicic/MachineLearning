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

    df_zeros = dataTrain.loc[dataTrain['isChina'] == 0]
    df_ones = dataTrain.loc[dataTrain['isChina'] == 1]

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

    actual_values = dataTest['isChina'].tolist()
    test_set = dataTest.values.tolist()

    return training_dict, test_set, actual_values


def fit(training_dict, alpha = 1.0):

    #ukupan broj instanci u dataset-u
    N = float(len(training_dict[0.0]) + len(training_dict[1.0]))

    #broj instanci po klasi
    Nc = [len(training_dict[key]) for key in training_dict]

    #prior_c = [float(c/N) for c in Nc]

    #verovatnoca za svaku klasu -> broj u toj klasi / ukupan broj
    class_probabilities_ = [1.0 * c / N for c in Nc]

    #brojimo koliko se puta koji atribut (rec) pojavio u kojoj klasi
    count = []
    for key in training_dict:
        count.append([sum(x) for x in zip(*training_dict[key])])

    #dodajemo smooting parametar na svaku vrednsost
    count_smooth = []
    for i in count:
        count_smooth.append(map(lambda x : x + alpha, i))

    #suma frekvencija svih reči u svim dokumentima koje pripadaju klasi c
    sum_Nc = []
    for c in count_smooth:
        sum_Nc.append(sum(c))

    #calculate the log probability of each word Tct
    #suma frekvencija reci (atributa) d svih instanci koji pridapaju klasi c






























