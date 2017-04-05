'''
    Function that splits dataset on training and validation validation sets.
'''
def split_dataset(x, y, percent):

    train_len = int(len(x) * percent)

    x_train = x[:train_len]
    y_train = y[:train_len]

    x_validate = x[train_len:]
    y_validate = y[train_len:]

    return x_train, y_train, x_validate, y_validate

'''
    Function that calculates RMSE.
'''
def calculate_rmse(actual, predicted):

    error_sum = 0.0
    n = len(actual)

    for i in range(n):
        error_sum += (predicted[i] - actual[i]) ** 2

    return (error_sum / n) ** 0.5


'''
    Function that predicts value of y for given x.
'''
def predict(row, coefficients, deg):
    prediction = coefficients[0]
    for i in range(deg):
        tmp_pow = i + 1
        prediction += coefficients[tmp_pow] * (row[0] ** tmp_pow)  # ... + cn * x^n ...  cn is nth coefficient
    
    return prediction


'''
    Function that estimates linear regression coefficients using stochastic gradient descent (SGD).

    Estimation until maximum number of iterations is reached or error between two iterations is
    small enough that it can be ignored.
'''
def estimate_coefficients(train, alpha=0.1, max_iters=1000, deg=1, epsilon=0.00001):
    coefs = [0.0 for i in range(deg + 1)]

    prev_error = 0
    iter_error = 0
    for iteration in range(max_iters):

        if iteration > 0 and abs(iter_error - prev_error) < epsilon:
            break

        prev_error = iter_error
        iter_error = 0

        for row in train:
            prediction = predict(row, coefs, deg)
            actual = row[1]
            error = prediction - actual
            iter_error += error ** 2
            coefs[0] -= alpha * error
            for i in range(deg):
                tmp_pow = i + 1
                gradient = row[0] ** tmp_pow  # derivate by theta coefficient by x on tmp_pow
                coefs[tmp_pow] -= alpha * error * gradient

    return coefs


'''
    Function that calculates predicitions for dataset.
'''
def calculate_predictions(dataset, coefs, deg):
    predictions = []

    for row in dataset:
        prediction = predict(row, coefs, deg)
        predictions.append(prediction)

    return predictions
