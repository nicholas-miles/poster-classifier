# pylint: disable=no-member
"""
URL and JSON tools for OMDB retrieval
BeautifulSoup for web scraping
"""
# file modification
import json
from uuid import uuid4
from shutil import copyfileobj
from contextlib import closing
# analysis toolkits
import pandas as pd
import numpy as np
from requests import get
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_api_key(filepath):
    """
    Retrieve the current OMDB API key, returns a string
    """
    with open(filepath, 'r') as api_file:
        return api_file.readline().replace('\n', '').replace('\r', '')


def is_good_response(resp, expected='html'):
    """
    Returns true if the response is the expected format, false otherwise
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 and
            content_type is not None and
            content_type.find(expected) > -1)


def simple_get(url, expected='html', payload=None):
    """
    Attempts to get the content at \\url\\ by making an HTTP GET request
    """
    try:
        with closing(get(url, stream=True, params=payload)) as resp:
            if is_good_response(resp, expected):
                return resp.text
            return None

    except RequestException as err_msg:
        print('Error during requests to {0} : {1}'.format(url, str(err_msg)))
        return None


def image_get(url, expected='image', payload=None, filename=str(uuid4())):
    """
    attempts to stream content at \\url\\ to image file
    """
    try:
        with closing(get(url, stream=True, params=payload)) as resp:
            if is_good_response(resp, expected):
                ext_pos = url.rfind('.')
                if ext_pos == -1:
                    return_format = 'jpg'
                else:
                    return_format = url[url.rfind('.')+1:]

                filepath = '../data/posters/' + filename + '.' + return_format
                with open(filepath, 'wb') as out_file:
                    copyfileobj(resp.raw, out_file)

                return filepath
            return None

    except RequestException as err_msg:
        if url == 'N/A':
            pass
        else:
            print('Error during requests to {0} : {1}'.format(url, str(err_msg)))
            return None


def imdb_titles(params):
    """
    Scrapes imdb IDs from top titles page
    returns list of imdb IDs
    """
    html_result = simple_get(
        'http://www.imdb.com/search/title', payload=params)

    soup = BeautifulSoup(html_result, 'html.parser')

    id_l = []

    for struct in soup.findAll('h3', {'class': 'lister-item-header'}):
        id_content = struct.find('a')['href']
        res = id_content.split('/')[2]

        id_l.append(res)

    return id_l


def get_omdb_data(imdb_id):
    """
    Retrieve data from OMDB given parameter set, returns a JSON file
    """
    url = 'http://www.omdbapi.com/'
    payload = {
        'apikey': get_api_key('../omdb_api_key.txt'),
        'i': imdb_id,
        'plot': 'short'
    }

    raw_json = simple_get(url, 'json', payload)

    if raw_json is None:
        return None

    return json.loads(raw_json)


if __name__ == '__main__':
    CONTENT = 5000
    TYPE = 'movies'
    PICKLE_NAME = 'movie_data'

    OMDB_DATA = []

    for i in tqdm(range(1, CONTENT // 50 + 1)):
        search_param = {'title_type': TYPE, 'page': str(i)}

        for val in tqdm(imdb_titles(search_param)):
            omdb = get_omdb_data(val)
            if omdb is not None and omdb['Poster'] != 'nan':
                OMDB_DATA.append(omdb)

    df = pd.DataFrame(OMDB_DATA)[['imdbID', 'Poster', 'Genre']]
    df['Genre'] = df.Genre.apply(lambda x: str(x).split(', '))
    df['Filepath'] = np.vectorize(image_get)(url=df['Poster'], filename=df['imdbID'])

    df.set_index('imdbID')
    df.to_pickle('../data/out/{}.pkl'.format(PICKLE_NAME))
