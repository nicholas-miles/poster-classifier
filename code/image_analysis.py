from PIL import Image
import os


def read_image_loc(fp):
    directory = os.fsencode(fp)
    for file in os.listdir(directory):
        infile = Image.open(fp + file.decode('utf-8'))
        print(infile.getbands())


read_image_loc('../data/posters/')
