from math import sqrt

from collections import namedtuple
Point_Center = namedtuple('Point', ('coords'))


def cluster_points(X, centers):
    clusters = {}

    for x in X:
        min_distance = float('inf')
        cluster_key = 0
        for i in range(len(centers)):
            tmp_center = centers[i]
            distance = euclidean(x, tmp_center)
            if distance < min_distance:
                min_distance = distance
                cluster_key = i

        try:
            clusters[cluster_key].append(x)
        except KeyError:
            clusters[cluster_key] = [x]

    return clusters


def euclidean(p1, p2):
    return sqrt(sum([(p1.coords[i] - p2.coords[i]) ** 2 for i in range(len(p1.coords))]))


def reevaluate_centers(clusters):
    new_centers = []
    keys = sorted(clusters.keys())

    for k in keys:
        sum_r = 0
        sum_g = 0
        sum_b = 0
        cluster = clusters[k]

        for point in cluster:
            sum_r += point.coords[0]
            sum_g += point.coords[1]
            sum_b += point.coords[2]

        n = len(cluster)
        new_center = (sum_r/n, sum_g/n, sum_b/n)
        new_centers.append(Point_Center(new_center))

    return new_centers


def has_converged(tmp_c, old_c):
    return (set([tuple(c1.coords) for c1 in tmp_c]) == set([tuple(c2.coords) for c2 in old_c]))


def find_centers(X, K, tmp_centers, max_iters):

    old_centers = tmp_centers
    iter_num = 0
    while iter_num == 0 or not has_converged(tmp_centers, old_centers):

        old_centers = tmp_centers

        # Assign all points in X to clusters
        clusters = cluster_points(X, tmp_centers)

        # Reevaluate centers
        tmp_centers = reevaluate_centers(clusters)
        iter_num += 1
        if iter_num >= max_iters:
            break

    return tmp_centers, clusters
