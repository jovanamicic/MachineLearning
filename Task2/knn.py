# Example of kNN implemented from Scratch in Python

import math
import operator
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

                    
def loadDataset(trainFileName, testFileName):
    dataTrain = pd.read_csv(trainFileName, header=0)
    trainSetY = dataTrain['Grade'].values.tolist()
    
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
    
    trainingSet = dataTrain.values.tolist()
    trainSetFeatures = dataTrain.drop('Grade', axis=1).values.tolist()
                
        
    dataTest = pd.read_csv(testFileName, header=0)
    testSetY = dataTest['Grade'].values.tolist()
    
    dataTest['sex'] = dataTest['sex'].map({'F': 0, 'M': 1})
    dataTest['address'] = dataTest['address'].map({'U': 0, 'R': 1})
    dataTest['famsize'] = dataTest['famsize'].map({'LE3': 0, 'GT3': 1})
    dataTest['Pstatus'] = dataTest['Pstatus'].map({'T': 0, 'A': 1})
    dataTest['reason'] = dataTest['reason'].map({'home': 0, 'reputation': 1, 'course' : 2, 'other' : 3})
    dataTest['guardian'] = dataTest['guardian'].map({'mother': 0, 'father': 1, 'other' : 2})
    dataTest['schoolsup'] = dataTest['schoolsup'].map({'yes': 0, 'no': 1})
    dataTest['famsup'] = dataTest['famsup'].map({'yes': 0, 'no': 1})
    dataTest['paid'] = dataTest['paid'].map({'yes': 0, 'no': 1})
    dataTest['activities'] = dataTest['activities'].map({'yes': 0, 'no': 1})
    dataTest['higher'] = dataTest['higher'].map({'yes': 0, 'no': 1})
    dataTest['internet'] = dataTest['internet'].map({'yes': 0, 'no': 1})
    dataTest['romantic'] = dataTest['romantic'].map({'yes': 0, 'no': 1})
    
    testSet = dataTest.values.tolist()
    testSetFeatures = dataTest.drop('Grade', axis=1).values.tolist()
    
    return trainingSet, testSet, trainSetY, testSetY, trainSetFeatures, testSetFeatures
    
                

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def get_rmse(y, y_pred):
    return mean_squared_error(y, y_pred) ** 0.5

    
def main():
    # prepare data
    trainingSet, testSet, trainSetY, testSetY, trainSetFeatures, testSetFeatures = loadDataset('train.csv', 'test.csv')
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    print

    # generate predictions
    predictions=[]
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    
    rmse = get_rmse(testSetY, predictions)
    print 'RMSE: ', rmse
    
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    
    
    print
    model = KNeighborsRegressor(n_neighbors=5, metric='manhattan')
    model.fit(trainSetFeatures, trainSetY)
    new_values = model.predict(testSetFeatures)
    print 'RMSE with sklearn: ' , get_rmse(testSetY, new_values)
    
main()