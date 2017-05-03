# -*- coding: utf-8 -*-

import sys
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import svm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from SVM import *


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

    training_set_x, training_set_y, test_set, actual_values = load_dataset2(filename_train, filename_test)
    
    
    list_of_c = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    list_of_gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    prediction_validations = []

    for c in list_of_c:
        for g in list_of_gammas: 
            clf = svm.SVC(C = c, gamma = g, kernel='rbf', max_iter=-1).fit(training_set_x, training_set_y) 
            predicted = clf.predict(test_set)
            tuple_predict = (c, g, predicted)
            prediction_validations.append(tuple_predict)


    all_scores_dict = {}
    max_f1_score = 0
    for p in prediction_validations:
        f1_score_res = f1_score(actual_values, p[2], average='micro')
#        print('c: {0}, gamma : {1}').format(p[0], p[1])
#        print('F1 micro score: {0} %').format(f1_score_res)
#        print
        if f1_score_res in all_scores_dict:
            all_scores_dict[f1_score_res].append((p[0], p[1]))
        else:
            all_scores_dict.update({f1_score_res:[(p[0], p[1])]})
            
    
    
    max_key = max(float(k) for k in all_scores_dict)
    for s in all_scores_dict[max_key]:
        print('BEST c: {0}, gamma : {1}, score : {2} %').format(s[0], s[1], max_key)
    
        
#    clf = svm.SVC(C = c, gamma = g, kernel='rbf', max_iter=-1).fit(training_set_x, training_set_y) 
#    predicted = clf.predict(test_set)
#    f1_score_res = f1_score(actual_values, predicted, average='micro')
#    print f1_score_res
    
