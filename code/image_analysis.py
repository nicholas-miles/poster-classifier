from PIL import Image
import os
import operator


def read_image_loc(fp):
    images = []
    directory = os.fsencode(fp)

    for file in os.listdir(directory):
        images.append(Image.open(fp + file.decode('utf-8')))

    return images


def main_colour(im, ignored_param=5, top_colours=1):
    ignored_colours = []

    for r in range(0, ignored_param + 1):
        for g in range(0, ignored_param + 1):
            for b in range(0, ignored_param + 1):
                ignored_colours.append((r,g,b))

    for r in range(255 - ignored_param, 256):
        for g in range(255 - ignored_param, 256):
            for b in range(255 - ignored_param, 256):
                ignored_colours.append((r,g,b))

    d = dict((y, x) for x, y in im.convert('RGB').getcolors())

    for key, value in list(d.items()):
        if key in ignored_colours:
            del d[key]


    sort_list = list(sorted(d.items(), key=operator.itemgetter(1), reverse=True))

    return [x for x, y in sort_list][:top_colours]




image_list = read_image_loc('../data/posters/')

opt_image_list = [im.convert('P', palette=Image.ADAPTIVE) for im in image_list]

print(main_colour(opt_image_list[0],top_colours=5))
