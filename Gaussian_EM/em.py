import numpy as np
from scipy.stats import norm


def norm_probality(value, mean, std):

    # the probability density function
    # value - RGB values of each picture

    p = 1
    for i in range(len(value)):
        p *= norm.pdf(value[i], mean[i], std[i])

    return p


def expectation_maximization(data, clusters, num_of_iterations, apriori, means, std):

    old_means = means
    iter_num = 0
    while iter_num == 0 or not has_converged(means, old_means):

        clusters, apriori = find_clusters(data, clusters, apriori, means, std)

        old_means = means
        means, std = recalculate_parameters(clusters)

        iter_num += 1
        print("EM Iteration:", iter_num)
        if iter_num >= num_of_iterations:
            break

    return clusters


def find_clusters(data, clusters, apriori, means, variances):

    new_clusters = {}
    weights_of_cluster = {}
    for point in data:

        # --------------------------------- E STEP ---------------------------------

        # calculate weights
        weights = []
        cluster_probabilities = {}

        x2 = 0
        for cluster in clusters:
            x1 = apriori[cluster] * norm_probality(point.coords, means[cluster], variances[cluster])  # iznad razlomka
            cluster_probabilities[cluster] = x1
            x2 += apriori[cluster] * cluster_probabilities[cluster] # dole u razlomku
        for cluster in clusters:
            weights.append(cluster_probabilities[cluster] / x2)
            weights_of_cluster[cluster] = weights

        # --------------------------------- M STEP ---------------------------------

        # assign to cluster with max weight
        cluster_key = np.argmax(weights)

        try:
            new_clusters[cluster_key].append(point)
        except KeyError:
            new_clusters[cluster_key] = [point]

    apriori_new = []
    nk_soft = {}
    for cluster in clusters:
        nk_soft[cluster] = sum(weights_of_cluster[cluster])
        apriori_new.append(nk_soft[cluster] / len(data))

    return new_clusters, apriori_new


def recalculate_parameters(clusters):
    means = {}
    variances = {}

    for k in clusters:
        sum_r = 0
        sum_g = 0
        sum_b = 0
        var_r = 0
        var_g = 0
        var_b = 0

        cluster = clusters[k]

        for point in cluster:
            sum_r += point.coords[0]
            sum_g += point.coords[1]
            sum_b += point.coords[2]
            var_r += point.std[0]
            var_b += point.std[1]
            var_g += point.std[2]

        means[k] = [sum_r / len(cluster), sum_g / len(cluster), sum_b / len(cluster)]
        variances[k] = [var_r / len(cluster), var_g / len(cluster), var_b / len(cluster)]

    return means, variances


def recalculate_probability(clusters, data):

    # update apriori probability for clusters
    cluster_probabilities = []

    for cluster in clusters:
        cluster_probabilities.append((1.0 * len(clusters[cluster]) / len(data)) + 1)  # smoothing parameter

    return cluster_probabilities


def has_converged(old_means, means):

    for i in means:
        if set(means[i]) != set(old_means[i]):
            return False

    return True
