import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score


'''
    Function that load test and training set from CSV files.
    All columns with missing values are deleted.
    Function also returns train data set with calculate mean and std for every attribute.
'''
def load_dataset(filename_train, filename_test):
    #Train set
    dataTrain = pd.read_csv(filename_train, header=0)
    dataTrain = dataTrain.replace('?', np.nan)
    dataTrain = dataTrain.dropna()
    dataTrain  = dataTrain.astype(float)
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
    
    print training_set_mean_std

    #Test set
    dataTest = pd.read_csv(filename_test, header=0)
    dataTest = dataTest.replace('?', np.nan)
    dataTest = dataTest.dropna()
    dataTest  = dataTest.astype(float)
    actual_values = dataTest['contraceptive'].tolist()
    test_set = dataTest.values.tolist()
    
    return training_set, test_set, actual_values, training_set_mean_std
 
#def separateByClass(dataset):
#	separated = {}
#	for i in range(len(dataset)):
#		vector = dataset[i]
#		if (vector[-1] not in separated):
#			separated[vector[-1]] = []
#		separated[vector[-1]].append(vector)
#	return separated
#
#def mean(numbers):
#     return np.mean(numbers)
#
#def stdev(numbers):
#     return np.std(numbers)
#
#def summarize(dataset):
#      summaries = [(mean(attribute), stdev(attribute)) for attribute in (dataset)]
#      del summaries[-1]
#      return summaries

#def summarizeByClass(dataset, class_zero, class_one):
#    #separated = separateByClass(dataset)
#    summaries = {}
#    for classValue, instances in separated.iteritems():
#        summaries[classValue] = summarize(instances)
#    return summaries

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

