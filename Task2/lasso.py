import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from util import *

def main():

	#Load dataset
	dataTrain = pd.read_csv("train.csv", header=0)
	y = np.array(dataTrain['Grade'].values.tolist())
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
	del dataTrain['Grade']

	dataTrain = np.array(dataTrain)
	X = dataTrain
	

	# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
	clf = LassoCV()

	# Set a minimum threshold of 0.25
	sfm = SelectFromModel(clf, threshold=0.25)
	sfm.fit(X, y) 
	n_features = sfm.transform(X).shape[1]
	#print("AAAA", n_features)
	print sfm.coef_      
	#print(X[100])

	#print(sfm.transform(X)[100])

main()
