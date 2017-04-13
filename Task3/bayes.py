import pandas as pd
import numpy as np
import random
import math

def load_dataset(filename_train, filename_test):
    #Train set
    dataTrain = pd.read_csv(filename_train, header=0)
    dataTrain = dataTrain.replace('?', np.nan)
    dataTrain = dataTrain.dropna()
    dataTrain  = dataTrain.astype(float)
    trainingSet = dataTrain.values.tolist()

    #Test set
    
    dataTest = pd.read_csv(filename_test, header=0)
    dataTest[:] = dataTest.replace('?', np.nan)
    dataTest = dataTest.dropna()
    dataTest  = dataTest.astype(float)
    testSet = dataTest.values.tolist()

    return trainingSet, testSet

"""
Function that separate rows in map with keys 0 and 1.
"""
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

"""
For every column calculate mean and std and put it in dictionary with keys 0 and 1.
"""
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    print(summaries)
    return summaries

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

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            print("true")
            correct += 1
    return (correct/float(len(testSet))) * 100.0
