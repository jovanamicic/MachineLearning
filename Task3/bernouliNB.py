import numpy as np
import pandas as pd

np.set_printoptions(precision=6)


'''
    Function that load test and training set from CSV files.
    All columns with missing values are deleted.
    Function also returns train data set with calculate mean and std for every attribute.
'''
def load_dataset(filename_train, filename_test):

    # ------------------------ TRAIN SET ---------------------------
    dataTrain = pd.read_csv(filename_train, header=0)

    dataTrain = dataTrain.replace('?', np.nan)
    dataTrain = dataTrain.astype(float)

    #izbaci nedostajuca
    dataTrain = dataTrain.dropna()

    #? zameni sa mean te kolone
    #dataTrain = dataTrain.apply(lambda x: x.fillna(x.mean()), axis=0)

    train_y = dataTrain['contraceptive'].tolist()
    train_x = dataTrain.values.tolist()



    # ------------------------ TEST SET ---------------------------
    dataTest = pd.read_csv(filename_test, header=0)

    dataTest = dataTest.replace('?', np.nan)
    dataTest = dataTest.astype(float)

    #izbaci nedostajuce
    dataTest = dataTest.dropna()

    #zameni sa mean te kolone
    #dataTest = dataTest.apply(lambda x: x.fillna(x.mean()), axis=0)

    actual_values = dataTest['contraceptive'].tolist()
    test_set = dataTest.values.tolist()

    return train_x, train_y, test_set, actual_values

class BernoulliNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.class_log_prior_ = [np.log(1.0 * len(i) / count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        smoothing = 2 * self.alpha
        n_doc = np.array([len(i) + smoothing for i in separated])
        self.feature_prob_ = count / n_doc[np.newaxis].T
        return self

    # OVA FJA NE RADI!!!
    def predict_log_proba(self, X):
        return [(np.log(self.feature_prob_) * x +
                 np.log(1 - self.feature_prob_) * np.abs(x - 1)).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)