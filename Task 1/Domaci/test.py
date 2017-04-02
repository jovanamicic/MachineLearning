import sys
from linear_regression import *
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':

    if len(sys.argv) == 1:
        filename = 'data/test.csv'
    else:
        filename = sys.argv[1]

    headers = ['X', 'Y']
    data = pd.read_csv(filename, names=headers, header=1)
    x = data['X']
    y = data['Y']

    # 80% data is training set and 20% of data is validation set
    x_train, y_train, x_validate, y_validate = split_dataset(x, y, 0.8)

    deg = 5  # polynom degree
    lеаrning_rate = 0.1  # learning rate
    max_iters = 10000

    # TRAINING SET

    dataset_train = []
    for xi, yi in zip(x_train, y_train):
        dataset_train.append([xi, yi])

    coefs = estimate_coefficients(dataset_train, lеаrning_rate, max_iters, deg)

    predicted_y = calculate_predictions(dataset_train, coefs, deg)

    plt.title("Training")
    plt.plot(x_train, y_train, 'go')
    plt.plot(x_train, predicted_y, 'r*')
    plt.show()

    print('RMSE TRAIN: ', calculate_rmse(list(y_train), predicted_y))


    # VALIDATION SET

    dataset_val = []
    for xi, yi in zip(x_validate, y_validate):
        dataset_val.append([xi, yi])

    y_validate_prediction = calculate_predictions(dataset_val, coefs, deg)

    plt.title("Validation")
    plt.plot(x_validate, y_validate, 'go')
    plt.plot(x_validate, y_validate_prediction, 'r*')
    plt.show()

    print('RMSE VALIDATION: ', calculate_rmse(list(y_validate), y_validate_prediction))