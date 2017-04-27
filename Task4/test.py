# -*- coding: utf-8 -*-

import sys
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from SVM import load_dataset


if __name__ == '__main__':
    # run: python test.py
    if len(sys.argv) == 1: 
        filename_train = 'train.csv'
        filename_test = 'test.csv'
       
    # run: python test.py filename_test
    elif len(sys.argv) == 2:
        filename_train = 'train.csv'
        filename_test = sys.argv[1]

    # run: python test.py filename_train filename_test
    else:
        filename_train = sys.argv[1]
        filename_test = sys.argv[2]

    training_set_x, training_set_y, test_set, actual_values = load_dataset(filename_train, filename_test)
    
    clf = SVC()
    clf.fit(training_set_x, training_set_y) 
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    
    predicted = clf.predict(test_set)
    print(predicted)
    
    f1_score = f1_score(actual_values, predicted, average='micro')
    print('F1 micro score: {0} %').format(f1_score)

#    class_probabilities, feature_probabilities = fit(training_dict)
#    predictions = predict(test_set, class_probabilities, feature_probabilities)
#
#    accuracy = accuracy_score(actual_values, predictions) * 100
#    print('Accuracy: {0} %').format(accuracy)

