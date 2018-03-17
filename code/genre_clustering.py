import os
import pandas as pd
import numpy as np
from omdb_scraper import get_omdb_data
from image_analysis import *
from tqdm import tqdm
from collections import Counter
from sklearn.decomposition import PCA

if __name__ == '__main__':
    df = pd.DataFrame(os.listdir('../data/posters')[:200], columns=['filenames'])

    df['imdb_id'] = df.filenames.map(lambda x: ''.join(x.split('.')[:-1]))
    df['filepath'] = '../data/posters/' + df.filenames
    df['genre'] = df.imdb_id.map(lambda x: get_omdb_data(imdb_id=x)['Genre'])
    df['flat_image'] = df.filepath.map(lambda x: image_dict(x, 20, 10)['flat_image'])

    pixels = np.vstack(df.flat_image.values)

    km, _ = cluster_image(pixels, 16)

    np_tup = np.asarray(km.cluster_centers_, dtype='uint8').reshape((4,4,3))

    plt.figure()
    plt.imshow(np_tup)
    plt.show()

    new_pixel = np.argmin(km.transform(pixels), axis=1)

    df['closest_colour'] = df.flat_image.map(lambda x: np.argmin(km.transform(x), axis=1))
    df['freq'] = df.closest_colour.map(Counter)

    genre_df = df[['imdb_id','genre']]
    genre_df = genre_df.set_index(['imdb_id'])['genre'].apply(pd.Series).stack()
    genre_df = genre_df.reset_index()
    genre_df.columns = ['imdb_id','genre_id','genre']

    colour_df = pd.concat([df.imdb_id, df.freq.apply(pd.Series)], axis = 1).fillna(0.0)

    colour_freq = genre_df.merge(colour_df).drop(['genre_id'], axis = 1).groupby(['genre']).sum()

    pca = PCA(n_components=16)
    coefficients = pca.fit_transform(np.asarray(colour_freq))

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.show()

    colour_freq.to_csv(r'../data/out/colour_freq.csv', header=colour_freq.columns, index='genre', sep=',', mode='w')

    # print(res.head())

