import os
import pandas as pd
import numpy as np
from omdb_scraper import get_omdb_data
from image_analysis import*
from tqdm import tqdm
from collections import Counter

if __name__ == '__main__':
    df = pd.DataFrame(os.listdir('../data/posters')[:4], columns=['filenames'])

    df['imdb_id'] = df.filenames.map(lambda x: ''.join(x.split('.')[:-1]))
    df['filepath'] = '../data/posters/' + df.filenames
    df['genre'] = df.imdb_id.map(lambda x: get_omdb_data(imdb_id=x)['Genre'])
    df['flat_image'] = df.filepath.map(lambda x: image_dict(x, 20, 10)['flat_image'])

    pixels = np.vstack(df.flat_image.values)

    km, _ = cluster_image(pixels, 16)

    new_pixel = np.argmin(km.transform(pixels), axis=1)

    df['closest_colour'] = df.flat_image.map(lambda x: np.argmin(km.transform(x), axis=1))
    df['freq'] = df.closest_colour.map(Counter)
    print(df)

