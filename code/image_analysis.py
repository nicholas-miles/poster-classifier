#pylint: disable=invalid-name,no-member
import os
import random
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


def all_images(filepath, resize_height=100, num_image=None):
    images = []
    for file in os.listdir(filepath):
        image_file = cv2.imread(filepath + file)
        if image_file is not None:
            imdb_id = file[:file.find('.')]
            raw_image = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
            resized_image = resize(raw_image, height=resize_height)
            flat_image = flatten_image(resized_image)
            
            image_dict = {'imdb_id': imdb_id, 'image': resized_image, 'flat_image': flat_image}

            images.append(image_dict)

            if num_image is not None and num_image <= len(images):
                break

    return images


def random_image(filepath, resize_height=100):
    file = random.choice(os.listdir(filepath))

    imdb_id = file[:file.find('.')]
    raw_image = cv2.cvtColor(cv2.imread(filepath + file), cv2.COLOR_BGR2RGB)
    resized_image = resize(raw_image, height=resize_height)
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


def image_pca(flat_image_list, components=100):
    pixel_list = [x.flatten() for x in flat_image_list]
    image_array = np.vstack(pixel_list)
    print(image_array.shape)
    pca_obj = PCA(n_components = components)
    pca_obj.fit(image_array)
    print(pca_obj.explained_variance_ratio_)

    return pca_obj.transform(image_array)


if __name__ == '__main__':
    print('loading images...')
    ALL_IMAGES = all_images('../data/posters/', 20, 100)
    print('getting genre...')
    for image in ALL_IMAGES:
        image['genre'] = get_omdb_data(imdb_id = image['imdb_id'])['Genre']

    FLAT_IMAGES = [x['flat_image'] for x in ALL_IMAGES]

    transform_images = image_pca(FLAT_IMAGES)
    print(transform_images)
    raise SystemExit

    SELECTED_IMAGE = random_image('../data/posters/')

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
        COLOUR_BAR = plot_colors(COLOUR_HISTOGRAM,
                                 CLT.cluster_centers_)
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
