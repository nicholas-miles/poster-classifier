import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from image_analysis import *
from sklearn.decomposition import PCA
from time import time


class GenreEmbedding(object):
    def test(self):
        print("Loading Pickles")
        dfs = self.load_pickles()

        print("Clustering")
        km = self.train_k_means(dfs['image'])

        print("Creating histograms")
        df = self.create_histogram_table(km, dfs)

        print("Embedding")
        embedded = self.embed_features(df)

        return embedded

    def load_pickles(self):
        fpath_pkl = lambda x: '../data/out/{}_data.pkl'.format(x) 
        dfs = {x: pd.read_pickle(fpath_pkl(x)) for x in self.dataframes}
        return dfs

    def train_k_means(self, df: pd.DataFrame):  
        pixels = np.vstack(df.flat_image.values)
        km, _ = cluster_image(pixels, self.num_clusts)
        return km

    def create_histogram_table(self, km, dfs: dict) -> pd.DataFrame:
        dfs['image'] = self.add_pixel_freq_col(km, dfs['image'])

        # TODO: recache pandas datframe and remove this hook
        dfs['movie'].columns = ['imdb_id', 'poster', 'genre', 'filepath']

        dfs = self.explode_dataframes(dfs)
        df = self.join_into_pivot(dfs)
        df = self.normalize_by_row(df)

        return df
    
    @staticmethod
    def add_pixel_freq_col(km, df: pd.DataFrame) -> pd.DataFrame:    
        df['freq'] = df.flat_image.map(km.transform)\
                                  .map(lambda x: np.argmin(x, axis=1))\
                                  .map(Counter)
        return df

    def explode_dataframes(self, dfs: dict) -> dict:
        for name, target in zip(self.dataframes, ['freq', 'genre']):
            cols = ['imdb_id', target]
            dfs[name] = self.explode(dfs[name][cols], target=target)
        dfs['image'].columns = ['imdb_id','cluster_id','freq']
        dfs['movie'].columns = ['imdb_id','genre_id','genre']
        return dfs

    @staticmethod
    def explode(df: pd.DataFrame, target: str) -> pd.DataFrame:
        df = df.set_index([x for x in df.columns if x != target])[target]\
               .apply(pd.Series)\
               .stack()\
               .reset_index()
        return df

    @staticmethod
    def join_into_pivot(dfs: dict) -> pd.DataFrame:
        df = dfs['movie'].merge(dfs['image'], on='imdb_id')
        cols = ['genre', 'cluster_id', 'freq']
        df = df[cols].groupby(['genre','cluster_id']).sum().reset_index()\
                     .pivot(index='genre', columns='cluster_id', values='freq')\
                     .fillna(0)
        return df

    @staticmethod
    def normalize_by_row(df):
        return df.divide(df.sum(axis=1), axis=0)

    def embed_features(self, df, verbose=False):
        pca = PCA()
        coefficients = pca.fit_transform(df)
        opt_c = np.argmax(pca.explained_variance_ratio_.cumsum() > self.cutoff)
        reduced_coefficients = coefficients[:opt_c+1]

        if verbose:
            plt.figure()
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.savefig('../data/out/var.png')
            plt.clf()

        return reduced_coefficients

    def __init__(self):
        self.num_clusts = 8
        self.dataframes = ['image', 'movie']
        self.cutoff = 0.9


if __name__ == '__main__':
    GE = GenreEmbedding()
    df = GE.test()
    print(df)

    # df.to_csv('../data/out/colour_freq.csv', header=df.columns, index='genre')

    # pca_c = [np.dot(colour_freq, x) for x in coefficients]