# -*- coding: utf-8 -*-

import sys
import ds
import kmeans
import em

from sklearn.metrics import accuracy_score

from collections import namedtuple
try:
    import Image
except ImportError:
    from PIL import Image

Point = namedtuple('Point', ('coords', 'variance', 'std', 'img_index', 'img_name'))


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

    # ------------------------------ K-MEANS ------------------------------------------------------
    dataset, dataset_means, dataset_variances, dataset_std, dataset_names = ds.load_dataset("data/*.jpg")
    points = []
    for i in range(len(dataset)):
        avg_color = (dataset_means[i][0], dataset_means[i][1], dataset_means[i][2])
        points.append(Point(avg_color, dataset_variances[i], dataset_std[i], i, dataset_names[i]))

    train, train_means, train_variances, train_std, train_names = ds.load_dataset("train/*.jpg")
    points_train = []
    for i in range(len(train)):
        avg_color_train = (train_means[i][0], train_means[i][1], train_means[i][2])
        points_train.append(Point(avg_color_train, train_variances[i], train_std[i], i, dataset_names[i]))

    initial_centeroids, initial_clusters = kmeans.find_centers(points, 4, points_train, 1000)
    # ---------------------------------------------------------------------------------------------

    # ------------------------------------------ EM -----------------------------------------------
    apriori = [0.25, 0.25, 0.25, 0.25]
    apriori = []

    for cluster in initial_clusters:
        apriori.append((1.0 * len(initial_clusters[cluster]) / len(points)) + 1)  # smoothing parameter


    means = {}
    std = {}

    for k in initial_clusters:
        sum_r = 0
        sum_g = 0
        sum_b = 0
        var_r = 0
        var_g = 0
        var_b = 0

        cluster = initial_clusters[k]

        for point in cluster:
            sum_r += point.coords[0]
            sum_g += point.coords[1]
            sum_b += point.coords[2]
            var_r += point.std[0]
            var_b += point.std[1]
            var_g += point.std[2]

        means[k] = [sum_r/len(cluster), sum_g/len(cluster), sum_b/len(cluster)]
        std[k] = [var_r / len(cluster), var_g / len(cluster), var_b / len(cluster)]

    clusters = em.expectation_maximization(points, initial_clusters, 100, apriori, means, std)
    # ---------------------------------------------------------------------------------------------

    for i in clusters:
        print(len(clusters[i]))
    # ---------------------------- VALIDATION -----------------------------------------------------
    names, actual_values = ds.load_test(filename_test)

    predictions = ds.get_predictions(clusters, names)

    accuracy = accuracy_score(actual_values, predictions) * 100
    print("ACCURACY:", accuracy)
