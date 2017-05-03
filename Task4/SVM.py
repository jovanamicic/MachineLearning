# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=6)


'''
    Function that load test and training set from CSV files.
    All columns with missing values are deleted.
    Function return training dataset separated by classes, and it returns
    test features and test actual values.
'''
def load_dataset2(filename_train, filename_test):
    
    # ------------------------ TRAIN SET ---------------------------
    dataTrain = pd.read_csv(filename_train, header=0)
    
    dataTrain = dataTrain.replace('?', np.nan)
    dataTrain = dataTrain.astype(float)

    # izbaci nedostajuca
    #dataTrain = dataTrain.dropna()

    # zameni nedostajuce sa mean te kolone
    dataTrain = dataTrain.apply(lambda x: x.fillna(x.mean()), axis=0)
    
    # izbaci irelevantne vrenosti
    #izbaciti: adhesion,epithelial,chromatin,mitoses
    dataTrain = dataTrain.drop('adhesion',1)
    dataTrain = dataTrain.drop('shape',1)
    dataTrain = dataTrain.drop('epithelial',1)
    dataTrain = dataTrain.drop('chromatin',1)
    
    #shufle dataset
    dataTrain = shuffle(dataTrain)
    

    data_set_y = dataTrain['class'].tolist()
    dataTrain = dataTrain.drop('class', 1)
    data_set_x = dataTrain.values.tolist()

    #podeli dataset
    training_set_x, test_set, training_set_y, actual_values = train_test_split(data_set_x, data_set_y, test_size=0.2, random_state=0)
    
    return training_set_x, training_set_y, test_set, actual_values
    
    
def load_dataset(filename_train, filename_test):
    
    # ------------------------ TRAIN SET ---------------------------
    dataTrain = pd.read_csv(filename_train, header=0)
    
    dataTrain = dataTrain.replace('?', np.nan)
    dataTrain = dataTrain.astype(float)

    # izbaci nedostajuca
    #dataTrain = dataTrain.dropna()

    # zameni nedostajuce sa mean te kolone
    dataTrain = dataTrain.apply(lambda x: x.fillna(x.mean()), axis=0)
    
    # izbaci irelevantne vrenosti, clump se ne sme izbaciti
    dataTrain = dataTrain.drop('adhesion',1)
    dataTrain = dataTrain.drop('shape',1)
    dataTrain = dataTrain.drop('epithelial',1)
    dataTrain = dataTrain.drop('chromatin',1)
    
    
    training_set_y = dataTrain['class'].tolist()
    dataTrain = dataTrain.drop('class', 1)
    training_set_x = dataTrain.values.tolist()


    # ------------------------ TEST SET ---------------------------
    dataTest = pd.read_csv(filename_test, header=0)

    dataTest = dataTest.replace('?', np.nan)
    dataTest = dataTest.astype(float)

    # izbaci nedostajuce
    #dataTest = dataTest.dropna()

    # zameni nedostajuce sa mean te kolone
    dataTest = dataTest.apply(lambda x: x.fillna(x.mean()), axis=0)
    
    # izbaci irelevantne vrenosti
    dataTest = dataTest.drop('adhesion',1)
    dataTest = dataTest.drop('shape',1)
    dataTest = dataTest.drop('epithelial',1)
    dataTest = dataTest.drop('chromatin',1)
    
    
    actual_values = dataTest['class'].tolist()
    test_set = dataTest.drop('class', 1).values.tolist()
    return training_set_x, training_set_y, test_set, actual_values
