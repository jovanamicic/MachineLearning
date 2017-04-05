import pandas as pd

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
