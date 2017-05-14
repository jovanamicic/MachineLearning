import numpy as np
import glob
from scipy import misc
import matplotlib.pyplot as plt
import pandas as pd

try:
    import Image
except ImportError:
    from PIL import Image


def load_dataset(dir_name):

    dataset = []
    dataset_means = []
    dataset_variances = []
    dataset_std = []
    dataset_names = []

    img_files = sorted(glob.glob(dir_name))
    for file in img_files:

        img = misc.imread(file).astype(float)

        dataset.append(img)
        mean_array = np.mean(img, axis=(0, 1))
        dataset_means.append(mean_array)

        variance_array = np.var(img, axis=(0, 1))
        dataset_variances.append(variance_array)

        std_array = np.std(img, axis=(0, 1))
        dataset_std.append(std_array)

        name = file.split("\\")[-1]
        dataset_names.append(name)

    return dataset, dataset_means, dataset_variances, dataset_std, dataset_names

def load_test(file):
    data = pd.read_csv(file, header=0)

    data = data.replace('cloudy_sky', 0)
    data = data.replace('rivers', 1)
    data = data.replace('sunsets', 2)
    data = data.replace('trees_and_forests', 3)

    names = data['image'].tolist()
    clusters = data['cluster'].tolist()
    return names, clusters

def get_predictions(clusters, names):

    predictions = []

    for img_name in names:
        for i in range(len(clusters)):
            cluster = clusters[i]
            for point in cluster:
                if point.img_name == img_name:
                    predictions.append(i)
                    break

    return predictions


def show_image(img, title):
    img /= 255
    plt.imshow(img, interpolation='nearest')
    plt.title(title)
    plt.show()
