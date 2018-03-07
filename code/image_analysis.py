import os
import operator
import random
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import cv2


def read_image_loc(filepath):
    images = []

    for file in os.listdir(filepath):
        images.append(cv2.cvtColor(cv2.imread(filepath + file), cv2.COLOR_BGR2RGB))

    return images


def main_colour(image_file, ignored_param=5, top_colours=1):
    ignored_colours = []

    for red in range(0, ignored_param + 1):
        for green in range(0, ignored_param + 1):
            for blue in range(0, ignored_param + 1):
                ignored_colours.append((red, green, blue))

    for red in range(255 - ignored_param, 256):
        for green in range(255 - ignored_param, 256):
            for blue in range(255 - ignored_param, 256):
                ignored_colours.append((red, green, blue))

    colour_dict = dict((y, x) for x, y in image_file.convert('RGB').getcolors())

    for key in list(colour_dict):
        if key in ignored_colours:
            del colour_dict[key]


    top_colour_list = list(sorted(colour_dict.items(), key=operator.itemgetter(1), reverse=True))

    return [x for x, y in top_colour_list][:top_colours]


def colour_distance(colour1, colour2):
    red_mean = (colour1[0] + colour2[0]) / 2
    red = colour1[0] - colour2[0]
    green = colour1[1] - colour2[1]
    blue = colour1[2] - colour2[2]

    return sqrt(((2 + (red_mean / 256)) * red * red) + (4 * (green * green)) + ((2 + ((255 - red_mean)/256)) * blue * blue))


def centroid_histogram(clt):
    num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, edges) = np.histogram(clt.labels_, bins = num_labels)

    hist = hist.astype('float')
    hist /= hist.sum()

    return hist


def plot_colors(hist, centroids):
    bar = np.zeros((50,300,3), dtype = 'uint8')
    startX = 0

    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype('uint8').tolist(), -1)
        startX = endX

    return bar


def cluster_image(image_file, clusters=20):
    image_arr = image_file.reshape((image_file.shape[0] * image_file.shape[1], 3))

    clt = KMeans(n_clusters = clusters)
    clt.fit(image_arr)

    return clt


IMAGE_LIST = read_image_loc('../data/posters/')
SELECTED_IMAGE = random.choice(IMAGE_LIST)

plt.figure()
plt.imshow(SELECTED_IMAGE)
plt.axis('off')

IMAGE_CLT = cluster_image(SELECTED_IMAGE, 5)
COLOUR_HISTOGRAM = centroid_histogram(IMAGE_CLT)
COLOUR_BAR = plot_colors(COLOUR_HISTOGRAM, IMAGE_CLT.cluster_centers_)

plt.figure()
plt.axis('off')
plt.imshow(COLOUR_BAR)
plt.show()

# TOP_COLOUR_LIST = main_colour(SELECTED_IMAGE, top_colours=5)

# for i in TOP_COLOUR_LIST:
#     for j in TOP_COLOUR_LIST:
#         print('Distance between {} and {} is {}'.format(i,j,colour_distance(i, j)))
