import os
import operator
import random
from PIL import Image
from math import sqrt


def read_image_loc(filepath):
    images = []
    directory = os.fsencode(filepath)

    for file in os.listdir(directory):
        images.append(Image.open(filepath + file.decode('utf-8')))

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


IMAGE_LIST = read_image_loc('../data/posters/')

OPT_IMAGE_LIST = [im.convert('P', palette=Image.ADAPTIVE) for im in IMAGE_LIST]

SELECTED_IMAGE = random.choice(OPT_IMAGE_LIST)
SELECTED_IMAGE.show()

TOP_COLOUR_LIST = main_colour(SELECTED_IMAGE, top_colours=5)

for i in TOP_COLOUR_LIST:
    for j in TOP_COLOUR_LIST:
        print('Distance between {} and {} is {}'.format(i,j,colour_distance(i, j)))
