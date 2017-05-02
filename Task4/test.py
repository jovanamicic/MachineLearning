# -*- coding: utf-8 -*-

import sys
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import svm

from SVM import load_dataset


if __name__ == '__main__':
	# run: python test.py
	if len(sys.argv) == 1: 
		filename_train = 'train2.csv'
		filename_test = 'test2.csv'
	   
	# run: python test.py filename_test
	elif len(sys.argv) == 2:
		filename_train = 'train.csv'
		filename_test = sys.argv[1]

	# run: python test.py filename_train filename_test
	else:
		filename_train = sys.argv[1]
		filename_test = sys.argv[2]

	training_set_x, training_set_y, test_set, actual_values = load_dataset(filename_train, filename_test)
	
	X = np.array(training_set_x).astype(int)
	y = training_set_y
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
	plt.show()

	"""
	#L1-based feature selection
	#http://scikit-learn.org/stable/modules/feature_selection.html
	X = np.array(training_set_x).astype(int)
	y = np.array(training_set_y).astype(int)
	lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
	model = SelectFromModel(lsvc, prefit=True)
	X_new = model.transform(X)
	print(X[0])
	print(X_new[0])
	#treba ostaviti: chromatin,size,shape,nuclei,nucleoli
	#izbaciti: adhesion,epithelial,clump,mitoses
	"""
	clf = SVC()
	clf.fit(training_set_x, training_set_y) 
	SVC(C=1.0, #C is 1 by default and it’s a reasonable default choice. If you have a lot of noisy observations you should decrease it.
		 cache_size=200,  #Kernel cache size -> It is recommended to set cache_size to a higher value than the default of 200(MB), such as 500(MB) or 1000(MB).
		 class_weight=None,
		 coef0=0.0,
			decision_function_shape=None,
		 degree=3,
		 gamma='auto', #The larger gamma is, the closer other examples must be to be affected.
		 kernel='rbf', # Radial Basis Function (RBF) kernel
			max_iter=-1,
		 probability=False,
		 random_state=None,
		 shrinking=True,
			tol=0.001,
		 verbose=False)

	predicted = clf.predict(test_set)
	"""
	C = 1.0  # SVM regularization parameter
	#clf = SVC(kernel='linear', C=C)
	#clf = SVC(kernel='rbf', gamma=0.7, C=C)
	#clf = SVC(kernel='poly', degree=3, C=C)
	clf = svm.LinearSVC(C=C)

	clf.fit(training_set_x, training_set_y) 
	predicted = clf.predict(test_set)
	"""
	f1_score = f1_score(actual_values, predicted, average='micro')
	print('F1 micro score: {0} %').format(f1_score)
