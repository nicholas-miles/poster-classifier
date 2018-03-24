import os
import pandas as pd
import numpy as np
from image_analysis import *
from collections import Counter
from sklearn.decomposition import PCA
from time import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    start_time = time()
    NUM_C = 4

    print('loading pickles')
    IM_DF = pd.read_pickle('../data/out/image_data.pkl')
    OMDB_DF = pd.read_pickle('../data/out/movie_data.pkl')
    pixels = np.vstack(IM_DF.flat_image.values)

    print('clustering')
    km, _ = cluster_image(pixels, NUM_C ** 2)
    np_tup = np.asarray(km.cluster_centers_, dtype='uint8').reshape((NUM_C, NUM_C, 3))

    print('dataframe manipulations')
    IM_DF['closest_colour'] = IM_DF.flat_image.map(lambda x: np.argmin(km.transform(x), axis=1))
    IM_DF['freq'] = IM_DF.closest_colour.map(Counter)

    genre_df = OMDB_DF[['imdbID','Genre']]
    genre_df = genre_df.set_index(['imdbID'])['Genre'].apply(pd.Series).stack()
    genre_df = genre_df.reset_index()
    genre_df.columns = ['imdb_id','genre_id','genre']

    colour_df = pd.concat([IM_DF.imdb_id, IM_DF.freq.apply(pd.Series)], axis = 1).fillna(0.0)

    colour_freq = genre_df.merge(colour_df).drop(['genre_id'], axis = 1).groupby(['genre']).sum()
    totals = pd.DataFrame(colour_freq.sum(axis = 1), columns = ['total'])
    colour_freq = pd.concat([colour_freq, totals], axis = 1)

    for n in range(NUM_C ** 2):
        colour_freq[n] = colour_freq[n] / colour_freq['total']

    print(colour_freq)

    print('Time taken is: ', time() - start_time)

    pca = PCA(n_components = NUM_C ** 2)
    features = [x for x in range(NUM_C ** 2)]
    coefficients = pca.fit_transform(colour_freq[features])

    print(coefficients)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.show()

    colour_freq.to_csv(r'../data/out/colour_freq.csv', header=colour_freq.columns, index='genre', sep=',', mode='w')

    # print(res.head())

