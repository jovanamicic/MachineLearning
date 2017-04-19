import math

import pandas as pd
import numpy as np


'''
    Function that load test and training set from CSV files.
    All columns with missing values are deleted.
    Function also returns train data set with calculate mean and std for every attribute.
'''
def load_dataset(filename_train, filename_test):
    #Train set
    dataTrain = pd.read_csv(filename_train, header=0)

    dataTrain = dataTrain.replace('?', np.nan)
    dataTrain = dataTrain.astype(float)

    #izbaci nedostajuca
    dataTrain = dataTrain.dropna()

    #? zameni sa mean te kolone
    #dataTrain = dataTrain.apply(lambda x: x.fillna(x.mean()), axis=0)

    training_set = dataTrain.values.tolist()

    training_set_mean_std = {0.0 : [], 1.0 : []}

    df_zeros = dataTrain.loc[dataTrain['contraceptive'] == 0]
    df_ones = dataTrain.loc[dataTrain['contraceptive'] == 1]

    for column in df_zeros.columns.values:
        if column != 'contraceptive':
            training_set_mean_std[0.0].append(tuple((df_zeros[column].mean(), df_zeros[column].std())))

    for column in df_ones.columns.values:
        if column != 'contraceptive':
            training_set_mean_std[1.0].append(tuple((df_ones[column].mean(), df_ones[column].std())))

    #print training_set_mean_std

    #Test set
    dataTest = pd.read_csv(filename_test, header=0)

    dataTest = dataTest.replace('?', np.nan)
    dataTest = dataTest.astype(float)

    #izbaci nedostajuce
    dataTest = dataTest.dropna()

    #zameni sa mean te kolone
    #dataTest = dataTest.apply(lambda x: x.fillna(x.mean()), axis=0)

    actual_values = dataTest['contraceptive'].tolist()
    test_set = dataTest.values.tolist()

    return training_set, test_set, actual_values, training_set_mean_std


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

 
'''
 Funkciji se prosledjuje test set i recnik ciji su kljuceci 0 i 1
 tj koristi ili ne koristi kontracepciju, a vrednosti su lista tuple-ova
 gde je prva vrednost u tuple-u mean, a druga std za svaki atribut.
 Ovo mozemo isto promeniti ako necemo da budu recnici.
'''
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

