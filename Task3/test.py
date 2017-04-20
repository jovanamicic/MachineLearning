import sys
from sklearn.metrics import accuracy_score

from multinomialNB import load_dataset, fit, predict


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

    training_dict, test_set, actual_values = load_dataset(filename_train, filename_test)

    class_probabilities, feature_probabilities = fit(training_dict)
    predictions = predict(test_set, class_probabilities, feature_probabilities)

    accuracy = accuracy_score(actual_values, predictions) * 100
    print('Accuracy: {0} %').format(accuracy)

