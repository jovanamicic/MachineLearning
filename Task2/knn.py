# Example of kNN implemented from Scratch in Python

import math
import operator
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from copy import deepcopy
from sklearn.metrics.pairwise import euclidean_distances

'''
    Function that reads data from train and test dataset.
    trainingSet, testSet - sets with all numeric attributes
    trainSetGrades, testSetGrades - sets with only data from 'Grade' column
    trainSetFeatures, testSetFeatures - sets with all attributes except 'Grade'
'''
#def load_dataset(trainFileName, testFileName):
#    dataTrain = pd.read_csv(trainFileName, header=0)
#    trainSetGrades = dataTrain['Grade'].values.tolist()
#    
#    dataTrain['sex'] = dataTrain['sex'].map({'F': 0, 'M': 1})
#    dataTrain['address'] = dataTrain['address'].map({'U': 0, 'R': 1})
#    dataTrain['famsize'] = dataTrain['famsize'].map({'LE3': 0, 'GT3': 1})
#    dataTrain['Pstatus'] = dataTrain['Pstatus'].map({'T': 0, 'A': 1})
#    dataTrain['reason'] = dataTrain['reason'].map({'home': 0, 'reputation': 1, 'course' : 2, 'other' : 3})
#    dataTrain['guardian'] = dataTrain['guardian'].map({'mother': 0, 'father': 1, 'other' : 2})
#    dataTrain['schoolsup'] = dataTrain['schoolsup'].map({'yes': 0, 'no': 1})
#    dataTrain['famsup'] = dataTrain['famsup'].map({'yes': 0, 'no': 1})
#    dataTrain['paid'] = dataTrain['paid'].map({'yes': 0, 'no': 1})
#    dataTrain['activities'] = dataTrain['activities'].map({'yes': 0, 'no': 1})
#    dataTrain['higher'] = dataTrain['higher'].map({'yes': 0, 'no': 1})
#    dataTrain['internet'] = dataTrain['internet'].map({'yes': 0, 'no': 1})
#    dataTrain['romantic'] = dataTrain['romantic'].map({'yes': 0, 'no': 1})
#    
#    dataTrain  = dataTrain.astype(float)
#    trainingSet = dataTrain.values.tolist()
#    trainSetFeatures = dataTrain.drop('Grade', axis=1).values.tolist()
#        
#    
#    
#    dataTest = pd.read_csv(testFileName, header=0)
#    testSetGrades = dataTest['Grade'].values.tolist()
#    
#    dataTest['sex'] = dataTest['sex'].map({'F': 0, 'M': 1})
#    dataTest['address'] = dataTest['address'].map({'U': 0, 'R': 1})
#    dataTest['famsize'] = dataTest['famsize'].map({'LE3': 0, 'GT3': 1})
#    dataTest['Pstatus'] = dataTest['Pstatus'].map({'T': 0, 'A': 1})
#    dataTest['reason'] = dataTest['reason'].map({'home': 0, 'reputation': 1, 'course' : 2, 'other' : 3})
#    dataTest['guardian'] = dataTest['guardian'].map({'mother': 0, 'father': 1, 'other' : 2})
#    dataTest['schoolsup'] = dataTest['schoolsup'].map({'yes': 0, 'no': 1})
#    dataTest['famsup'] = dataTest['famsup'].map({'yes': 0, 'no': 1})
#    dataTest['paid'] = dataTest['paid'].map({'yes': 0, 'no': 1})
#    dataTest['activities'] = dataTest['activities'].map({'yes': 0, 'no': 1})
#    dataTest['higher'] = dataTest['higher'].map({'yes': 0, 'no': 1})
#    dataTest['internet'] = dataTest['internet'].map({'yes': 0, 'no': 1})
#    dataTest['romantic'] = dataTest['romantic'].map({'yes': 0, 'no': 1})
#    
#    dataTest  = dataTest.astype(float)
#    testSet = dataTest.values.tolist()
#    testSetFeatures = dataTest.drop('Grade', axis=1).values.tolist()
#    
#    
#    return trainingSet, testSet, trainSetGrades, testSetGrades, trainSetFeatures, testSetFeatures
  
    
def load_dataset(trainFileName, testFileName):
    dataTrain = pd.read_csv(trainFileName, header=0)
    trainSetGrades = dataTrain['Grade'].values.tolist()[:400]
    
    dataTrain['sex'] = dataTrain['sex'].map({'F': 0, 'M': 1})
    dataTrain['address'] = dataTrain['address'].map({'U': 0, 'R': 1})
    dataTrain['famsize'] = dataTrain['famsize'].map({'LE3': 0, 'GT3': 1})
    dataTrain['Pstatus'] = dataTrain['Pstatus'].map({'T': 0, 'A': 1})
    dataTrain['reason'] = dataTrain['reason'].map({'home': 0, 'reputation': 1, 'course' : 2, 'other' : 3})
    dataTrain['guardian'] = dataTrain['guardian'].map({'mother': 0, 'father': 1, 'other' : 2})
    dataTrain['schoolsup'] = dataTrain['schoolsup'].map({'yes': 0, 'no': 1})
    dataTrain['famsup'] = dataTrain['famsup'].map({'yes': 0, 'no': 1})
    dataTrain['paid'] = dataTrain['paid'].map({'yes': 0, 'no': 1})
    dataTrain['activities'] = dataTrain['activities'].map({'yes': 0, 'no': 1})
    dataTrain['higher'] = dataTrain['higher'].map({'yes': 0, 'no': 1})
    dataTrain['internet'] = dataTrain['internet'].map({'yes': 0, 'no': 1})
    dataTrain['romantic'] = dataTrain['romantic'].map({'yes': 0, 'no': 1})
    
    dataTrain  = dataTrain.astype(float)
    
    dataTrainCopy = deepcopy(dataTrain)
    
    trainingSet = dataTrain.values.tolist()[:400]
    trainSetFeatures = dataTrain.drop('Grade', axis=1).values.tolist()[:400]
        
    
    
    #dataTest = pd.read_csv(testFileName, header=0)
    testSetGrades = dataTrainCopy['Grade'].values.tolist()[400:]
    
    testSet = dataTrainCopy.values.tolist()[400:]
    testSetFeatures = dataTrainCopy.drop('Grade', axis=1).values.tolist()[400:]
    
    
    return trainingSet, testSet, trainSetGrades, testSetGrades, trainSetFeatures, testSetFeatures
                
'''
    Function that calculates Euclidian distance between any two given instances.
'''
def euclidean_distance(testInstance, trainingInstance, length):
    return math.sqrt(sum(pow(a-b,2) for a, b in zip(testInstance, trainingInstance)))
#    distance = 0
#    for x in range(length):
#        distance += pow((testInstance[x] - trainingInstance[x]), 2)
#    return math.sqrt(distance)



    
def manhattan_distance(testInstance, trainingInstance, length):
    return sum(abs(a-b) for a,b in zip(testInstance,trainingInstance))
#    distance = 0
#    for x in range(length):
#         distance += abs(testInstance[x] - trainingInstance[x])
#    return distance
    
    
'''
    Function that return k most similar instances for given instance.
    It calculates similarity by calling euclidean_distance function.
    
    Funkciji se prosledjuje ceo trening set i samo jedna instaca testa,
    koja se poredi sa svakom instancom trening seta kako bi joj se naslo K najsliznijih tacaka
'''
def get_neighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        #dist = chebyshev_distance(testInstance, trainingSet[x], length)
        dist = euclidean_distance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
        
    return neighbors

'''
    After we find k nearest neighbors, we need to find a class
'''
def get_response(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def get_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def get_rmse(y, y_pred):
    return mean_squared_error(y, y_pred) ** 0.5

    
def main():
    # prepare data
    trainingSet, testSet, trainSetGrades, testSetGrades, trainSetFeatures, testSetFeatures = load_dataset('train.csv', 'test.csv')
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    print
    
#    meanTrainingSet = np.mean(trainingSet)
#    stdTrainingSet = np.std(trainingSet)
#    
#    meanTestSet = np.mean(testSet)
#    stdTestSet = np.std(testSet)
#    
#    trainingSetNor = (trainingSet - meanTrainingSet) / stdTrainingSet
#    testSetNor = (testSet - meanTestSet) / stdTestSet

    # generate predictions
    predictions=[]
    k = 7
    for x in range(len(testSet)):
        neighbors = get_neighbors(trainingSet, testSet[x], k)
        result = get_response(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

#    for x in range(len(testSet)):
#        neighbors = get_neighbors(trainingSetNor, testSetNor[x], k)
#        result = get_response(neighbors)
#        predictions.append(result)
#        print('> predicted=' + repr(result) + ', actual=' + repr(testSetNor[x][-1]))
   
    rmse = get_rmse(testSetGrades, predictions)
    print 'RMSE: ', rmse
    
    accuracy = get_accuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    
#    model = KNeighborsRegressor(n_neighbors=7, metric='euclidian')
#    model.fit(trainSetFeatures, trainSetGrades)
#    new_values = model.predict(testSetFeatures)
#    print 'RMSE with sklearn: ' , get_rmse(testSetGrades, new_values)
    
    
main()