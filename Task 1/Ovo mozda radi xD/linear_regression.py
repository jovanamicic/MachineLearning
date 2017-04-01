import matplotlib.pyplot as plt
import pandas as pd



def mean_normlisation():
    pass


def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return (mean_error) ** 0.5


# Make a prediction with coefficients
def predict(row, coefficients, deg):
    yhat = 0
    for i in range(deg):  # +1 zbog slobodnog clana???????
        yhat += coefficients[i] * row[0] ** i

    return yhat


# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch, deg=1):
    coef = [3.0 for i in range(deg)]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef, deg)
            error = yhat - row[-1]
            sum_error += error ** 2
            coef[0] -= l_rate * error
            for i in range(deg - 1):
                gd = row[0] ** i  # izvod po teta uz x na neki stepen
                coef[i + 1] -= l_rate * error * gd
                # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef


# MAIN

header = ['X', 'Y']
data = pd.read_csv('ships.csv', names=header, header=1)
x = data['X']
y = data['Y']

train_len = int(len(x) * 0.8)

x_train = x[:train_len]
y_train = y[:train_len]

x_validate = x[train_len:]
y_validate = y[train_len:]

deg = 5  # treba u fjama dodati svuda deg+1 zbog slobodnog clana
l_rate = 0.01
max_iters = 100


# TRENING SKUP

dataset = []
for xi, yi in zip(x_train, y_train):
    dataset.append([xi, yi])

coefs = coefficients_sgd(dataset, l_rate, max_iters, deg)
print(coefs)

predicted_y = []
for row in dataset:
    yhat = predict(row, coefs, deg)
    predicted_y.append(yhat)

plt.plot(x_train, y_train, 'go')
plt.plot(x_train, predicted_y, 'r*')
plt.show()

print('RMSE TRAIN: ', rmse_metric(list(y_train), predicted_y))


# VALIDACIONI SKUP

dataset_val = []
for xi, yi in zip(x_validate, y_validate):
    dataset_val.append([xi, yi])

y_validate_prediction = []
for row in dataset_val:
    yhat = predict(row, coefs, deg)
    y_validate_prediction.append(yhat)

plt.plot(x_validate, y_validate, 'go')
plt.plot(x_validate, y_validate_prediction, 'r*')
plt.show()

print('RMSE VALIDATION: ', rmse_metric(list(y_validate), y_validate_prediction))
