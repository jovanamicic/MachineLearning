import sys
import numpy as np
from sklearn.metrics import accuracy_score

from multinomialNB import *


if __name__ == '__main__':
    # run: python test.py
    if len(sys.argv) == 1: 
        filename_train = 'drzave.csv'
        filename_test = 'drzaveTest.csv'
       
    # run: python test.py filename_test
    elif len(sys.argv) == 2:
        filename_train = 'data/train.csv'
        filename_test = sys.argv[1]

    # run: python test.py filename_train filename_test
    else:
        filename_train = sys.argv[1]
        filename_test = sys.argv[2]

    # read dataset for gaussian NB
    # training_set, test_set, actual_values, training_set_mean_std = load_dataset(filename_train, filename_test)

    # predictions for gaussian NB
    #predictions = getPredictions(training_set_mean_std, test_set)


    # read dataset for multinomial/bernouli NB
    train_x, train_y, test_set, actual_values = load_dataset(filename_train, filename_test)

    model = MultinomialNB()
    model.fit(np.array(train_x), np.array(train_y))
    predictions = model.predict(test_set)


    accuracy = accuracy_score(actual_values, predictions) * 100
    print('Accuracy: {0} %').format(accuracy)

    #sa nasim: Accuracy: 99.5575221239 %
    #sa ugradjenim: Accuracy: 99.5575221239 %
