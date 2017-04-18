import sys
import pandas as pd
from bayes import *


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

    training_set, test_set, actual_values, training_set_mean_std = load_dataset(filename_train, filename_test)
    
    predictions = getPredictions(training_set_mean_std, test_set)
    accuracy = accuracy_score(actual_values, predictions) * 100
    print('Accuracy: {0} %').format(accuracy)