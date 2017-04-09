import math
import operator
import pandas as pd
from sklearn.metrics import mean_squared_error
from copy import deepcopy

'''
    Function that reads data from train and test dataset.
    trainingSet, testSet - sets with all numeric attributes
    trainSetGrades, testSetGrades - sets with only data from 'Grade' column
    trainSetFeatures, testSetFeatures - sets with all attributes except 'Grade'
'''
def load_dataset(trainFileName, testFileName):
    dataTrain = pd.read_csv(trainFileName, header=0)
    
    #list with only Y values from dataset
    trainSetGrades = dataTrain['Grade'].values.tolist()
    
    #transform to numerical
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
    
    #transform from int to float
    dataTrain  = dataTrain.astype(float)
    
    #drop unnecessary attributess from dataset
    #dataTrain = dataTrain.drop(['age', 'Fedu', 'reason', 'traveltime', 'famsup', 'famrel', 'freetime', 'goout', 'Walc', 'Medu', 'health', 'absences', 'guardian'], axis = 1)
    trainingSet = dataTrain.values.tolist()
    
    #list with only X attributes from train dataset
    trainSetFeatures = dataTrain.drop('Grade', axis=1).values.tolist()
        
    
    
    dataTest = pd.read_csv(testFileName, header=0)
    
    #list with only Y values from dataset
    testSetGrades = dataTest['Grade'].values.tolist()
    
    #transform to numerical
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
    
    #transform from int to float
    dataTest  = dataTest.astype(float)
    
    #drop unnecessary attributess from dataset
    dataTest = dataTest.drop(['age', 'Fedu', 'reason', 'traveltime', 'famsup', 'famrel', 'freetime', 'goout', 'Walc', 'Medu', 'health', 'absences', 'guardian'], axis = 1)
    testSet = dataTest.values.tolist()
    
    #list with only X attributes from test dataset
    testSetFeatures = dataTest.drop('Grade', axis=1).values.tolist()
    
    
    return trainingSet, testSet, trainSetGrades, testSetGrades, trainSetFeatures, testSetFeatures

  
    
def load_dataset2(trainFileName, testFileName):
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
    #dataTrain = dataTrain.drop(['age', 'Fedu', 'reason', 'traveltime', 'famsup', 'famrel', 'freetime', 'goout', 'Walc', 'Medu', 'health', 'absences', 'guardian'], axis = 1)
    
    
    
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


    
def manhattan_distance(testInstance, trainingInstance, length):
    return sum(abs(a-b) for a,b in zip(testInstance,trainingInstance))
    
    
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
        #dist = manhattan_distance(testInstance, trainingSet[x], length)
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
def predict_grade(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def get_rmse(y, y_pred):
    return mean_squared_error(y, y_pred) ** 0.5

    