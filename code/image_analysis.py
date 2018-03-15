#pylint: disable=invalid-name,no-member
import os
import random
import seaborn as sns
import time
from tqdm import tqdm
from math import sqrt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import cv2
from imutils import resize
from omdb_scraper import get_omdb_data


def flatten_image(image_file):
    return image_file.reshape((image_file.shape[0] * image_file.shape[1], 3))


def all_images(filepath, resize_height=100, resize_width=50, num_image=None):
    images = []
    for file in tqdm(os.listdir(filepath)):
        image_file = cv2.imread(filepath + file)
        if image_file is not None:
            imdb_id = file[:file.find('.')]
            raw_image = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(raw_image, (resize_width, resize_height))
            flat_image = flatten_image(resized_image)
            
            image_dict = {'imdb_id': imdb_id, 'image': resized_image, 'flat_image': flat_image}

            images.append(image_dict)

            if num_image is not None and num_image <= len(images):
                break

    return images


def image_dict(file, resize_height=100, resize_width=50):
    imdb_id = file[:file.find('.')]
    raw_image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(raw_image, (resize_width, resize_height))
    flat_image = flatten_image(resized_image)
    
    image_dict = {'imdb_id': imdb_id, 'image': resized_image, 'flat_image': flat_image}

    return image_dict


def colour_distance(colour1, colour2):
    red_mean = (colour1[0] + colour2[0]) / 2
    red = colour1[0] - colour2[0]
    green = colour1[1] - colour2[1]
    blue = colour1[2] - colour2[2]

    red_val = ((2 + (red_mean / 256)) * red * red)
    green_val = (4 * (green * green))
    blue_val = ((2 + ((255 - red_mean)/256)) * blue * blue)

    total_val = sqrt(red_val + green_val + blue_val)

    return total_val


def centroid_histogram(clt):
    num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=num_labels)

    hist = hist.astype('float')
    hist /= hist.sum()

    return hist


def plot_colors(hist, centroids):
    bar_plot = np.zeros((50, 300, 3), dtype='uint8')
    startX = 0

    for (percent, color) in sorted(zip(hist, centroids), key=lambda x: -x[0]):
        endX = startX + (percent * 300)
        start_coor = (int(startX), 0)
        end_coor = (int(endX), 50)
        cv2.rectangle(bar_plot, start_coor, end_coor, color.astype('uint8').tolist(), -1)
        startX = endX

    return bar_plot


def cluster_image(image_file, clusters=5):
    clt = KMeans(n_clusters=clusters)
    clt.fit(image_file)

    silhouettes = silhouette_score(image_file, clt.labels_, metric='euclidean', sample_size=300)

    return clt, silhouettes


def image_pca(images, components=200, comp_threshold=0.9):
    """Image PCA compression.

    Thnk Taylor series approximation in the case of approxamating a polynomial
    and we obtain say [a, b, c] for:
    y = a*x^0 + b*x^1 + c*x^2

    by analogy for our problem we have some components with coefficients
    [a, b, c] and matrices sorted by importance of eigen-values:
    img = a*X_0 + b*X_1 + c*X_2

    thus the approximate image is the inverse transform of PCA while the eigen
    basis is to be reshaped for the eigen images, because they are used to
    approximate the image there are no restrictions on values they take hold of
    thus we split their visualization into 3 channels for R, G and B
    """

    # reshape array of images into flattened representation
    image_array = images.reshape(len(images), -1)

    # perform PCA transform to compress images
    pca = PCA(n_components=components)

    # calculate coefficients
    coefficients = pca.fit_transform(image_array)

    # get the explained variance
    comp_variance_cum = np.cumsum(pca.explained_variance_ratio_)

    # get the eigen basis and reshape to see as eigen images
    eigen_image = pca.components_.reshape(-1, *images.shape[1:])

    # using the inverse transform (by analogy using the taylor approximation)
    # reconstrute the images
    projected = pca.inverse_transform(coefficients)\
                   .reshape(*images.shape)\
                   .astype(np.uint8)

    # visualize the new images in a new basis
    print_projected_images(projected)

    # visualize the matrices/flattened arrays/eigen images that were used to
    # construct the above images
    print_eigen_images(eigen_image, top_n=10)

    return projected


def print_projected_images(projected):
    fig = plt.figure()
    for y in range(5):
        for x in range(5):
            idx = 5*y + x+ 1
            ax1 = fig.add_subplot(5, 5, idx)
            plt.imshow(projected[idx, :, :, :])
            plt.xticks([])
            plt.yticks([])
    plt.show()


def print_eigen_images(eigen_image, top_n=5):
    fig = plt.figure()
    for y in range(top_n):
        for x in range(3):
            ax1 = fig.add_subplot(3, top_n, top_n*x + y + 1)
            sns.heatmap(eigen_image[y, :, :, x], cbar=False, cmap='jet')
            plt.xticks([])
            plt.yticks([])
    plt.show()

if __name__ == '__main__':

    IM_H_LIM = 50
    IM_W_LIM = 30

    print('loading images...')
    ALL_IMAGES = all_images('../data/posters/', IM_H_LIM, IM_W_LIM)

    # print('getting genre...')
    # for ix, image in enumerate(ALL_IMAGES):
    #     try:
    #         image['genre'] = get_omdb_data(imdb_id = image['imdb_id'])['Genre']
    #     except TypeError:
    #         del ALL_IMAGES[ix]

    images = np.array([x['image'] for x in ALL_IMAGES])
    pca_images = image_pca(images, 30)
    raise SystemExit

    # for i in pca_images:
    #     cv2.imshow(i.reshape(IM_H_LIM, IM_W_LIM, 3))

    # print(num_c)



    SELECTED_IMAGE = image_dict('../data/posters/')

    max_score = 0
    optimal_k = 0
    optimal_clt = None

    colour_bars = []
    for cluster_size in range(2, 17):
        CLT, SILHOUETTE_SCORE = cluster_image(SELECTED_IMAGE['flat_image'], cluster_size)
        if SILHOUETTE_SCORE > max_score:
            max_score = SILHOUETTE_SCORE
            optimal_k = cluster_size
            optimal_clt = CLT

        COLOUR_HISTOGRAM = centroid_histogram(CLT)
        COLOUR_BAR = plot_colors(COLOUR_HISTOGRAM, CLT.cluster_centers_)
        colour_bars.append(COLOUR_BAR)

    IMAGE_CLT = optimal_clt
    print(optimal_k)
    print(max_score)

    cbars = np.vstack(colour_bars)

    COLOUR_HISTOGRAM = centroid_histogram(IMAGE_CLT)
    COLOUR_BAR = plot_colors(COLOUR_HISTOGRAM, IMAGE_CLT.cluster_centers_)

    plt.figure()
    plt.imshow(SELECTED_IMAGE['image'])
    plt.axis('off')

    plt.figure()
    plt.axis('off')
    plt.imshow(COLOUR_BAR)

    plt.figure()
    plt.axis('off')
    plt.imshow(cbars)
    plt.show()

    # TOP_COLOUR_LIST = main_colour(SELECTED_IMAGE, top_colours=5)

# for i in TOP_COLOUR_LIST:
#     for j in TOP_COLOUR_LIST:
#         print('Distance between {} and {} is {}'.format(i,j,colour_distance(i, j)))
