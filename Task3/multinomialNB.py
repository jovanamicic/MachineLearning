import numpy as np
import pandas as pd

np.set_printoptions(precision=6)


'''
    Function that load test and training set from CSV files.
    All columns with missing values are deleted.
    Function return training dataset separated by classes, and it returns
    test features and test actual values.
'''
def load_dataset(filename_train, filename_test):

    # ------------------------ TRAIN SET ---------------------------
    dataTrain = pd.read_csv(filename_train, header=0)

    dataTrain = dataTrain.replace('?', np.nan)
    dataTrain = dataTrain.astype(float)

    # izbaci nedostajuca
    dataTrain = dataTrain.dropna()

    # zameni nedostajuce sa mean te kolone
    # dataTrain = dataTrain.apply(lambda x: x.fillna(x.mean()), axis=0)

    training_dict = {0.0: [], 1.0: []}

    df_zeros = dataTrain.loc[dataTrain['contraceptive'] == 0]
    df_zeros = df_zeros.drop('contraceptive', 1)
    df_ones = dataTrain.loc[dataTrain['contraceptive'] == 1]
    df_ones = df_ones.drop('contraceptive', 1)

    for row in  range(len(df_zeros.values.tolist())):
        training_dict[0.0].append(df_zeros.values.tolist()[row])

    for row in  range(len(df_ones.values.tolist())):
        training_dict[1.0].append(df_ones.values.tolist()[row])


    # ------------------------ TEST SET ---------------------------
    dataTest = pd.read_csv(filename_test, header=0)

    dataTest = dataTest.replace('?', np.nan)
    dataTest = dataTest.astype(float)

    # izbaci nedostajuce
    dataTest = dataTest.dropna()

    # zameni nedostajuce sa mean te kolone
    # dataTest = dataTest.apply(lambda x: x.fillna(x.mean()), axis=0)

    actual_values = dataTest['contraceptive'].tolist()
    test_set = dataTest.drop('contraceptive', 1).values.tolist()
    return training_dict, test_set, actual_values


'''
    Function that calculate probabilities based on data.
'''
def fit(training_dict, alpha=1.0):

    # total number of instances in training dataset
    N = float(len(training_dict[0.0]) + len(training_dict[1.0]))

    # total number of instances by classes
    Nc = [len(training_dict[key]) for key in training_dict]

    # probability for each class
    class_probabilities = [(1.0 * c / N) for c in Nc]

    # counting appearance of each attribute in class
    count = []
    for key in training_dict:
        count.append([sum(x) for x in zip(*training_dict[key])])

    # adding smoothing parameter on each value
    count_smooth = []
    for i in count:
        count_smooth.append(map(lambda x: x + alpha, i))

    # sum of frequencies of all words in all instances which belong to class c
    feature_probabilities = []
    for c in count_smooth:
        feature_probabilities.append((np.array(c) / sum(c)).tolist())

    return class_probabilities, feature_probabilities


'''
    Function that makes predictions for test data.
'''
def predict(test_set, class_probabilities, feature_probabilities):

    # making numpy arrays
    feature_probabilities = np.log(np.array(feature_probabilities))
    class_probabilities = np.log(np.array(class_probabilities))

    prediction = []
    for x in test_set:
        prediction.append((np.array(feature_probabilities) * np.array(x)).sum(axis=1) + class_probabilities)

    return np.argmax(prediction, axis=1)  # returns index of max element
