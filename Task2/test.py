import sys

from knn import load_dataset, find_k_neighbors, predict_grade, get_rmse


if __name__ == '__main__':

    # run: python test.py
    if len(sys.argv) == 1: 
        filename_train = 'data/train.csv'
        filename_test = 'data/test.csv'
       
    # run: python test.py filename_test
    elif len(sys.argv) == 2:
        filename_train = 'data/train.csv'
        filename_test = sys.argv[1]

    # run: python test.py filename_train filename_test
    else:
        filename_train = sys.argv[1]
        filename_test = sys.argv[2]


    trainingSet, testSet, trainSetGrades, testSetGrades, trainSetFeatures, testSetFeatures = load_dataset(filename_train, filename_test)
    
    predictions=[]
    k = 7
    for x in range(len(testSet)):
        neighbors = find_k_neighbors(trainingSet, testSet[x], k)
        predicted = predict_grade(neighbors)
        predictions.append(predicted)
        print'> predicted=' + str(predicted) + ', actual=' + str(testSetGrades[x])

    rmse = get_rmse(testSetGrades, predictions)
    print 'RMSE:', str(rmse)