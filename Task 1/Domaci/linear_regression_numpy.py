import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def read_file(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    x = data[:, 0] # prva kolona
    y = data[:, 1] # druga kolona
    return x, y


def calculate_rmse(actual, predicted):
    return sqrt((1.0 / predicted.size) * sum((actual - predicted) ** 2))


def linear_regression(x, y, deg):
    return np.polyfit(x, y, deg)

if __name__ == '__main__':
    x, y = read_file("ships.csv")

    #coefs = linear_regression(x, y, 4)
    coefs = [-4.0440399906047064, -4.0440399906047064, 132.37518934740396, 91.860493081239781, 50.42273333852679, 26.186050859633415]
    coefs = coefs[::-1]
    print(coefs)

    predicted = np.polyval(coefs, x)

    rmse = calculate_rmse(y, predicted)
    print("RMSE:", rmse)

    plt.plot(x, y, "rx")
    plt.plot(x, predicted, "bo")
    plt.show()


