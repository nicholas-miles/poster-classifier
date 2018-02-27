# pylint: disable=no-member
"""
URL and JSON tools for OMDB retrieval
BeautifulSoup for web scraping
pandas for data analysis
"""
from contextlib import closing
import json
from uuid import uuid4
import shutil
from requests import get
from requests.exceptions import RequestException
from bs4 import BeautifulSoup


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
                    return_format = 'png'
                else:
                    return_format = url[url.rfind('.')+1:]

                filepath = '../data/posters/' + filename + '.' + return_format
                with open(filepath, 'wb') as out_file:
                    shutil.copyfileobj(resp.raw, out_file)
            return None

    except RequestException as err_msg:
        if url == 'N/A':
            pass
        else:
            print('Error during requests to {0} : {1}'.format(url, str(err_msg)))
            return None


def is_good_response(resp, expected='html'):
    """
    Returns true if the response is the expected format, false otherwise
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 and
            content_type is not None and
            content_type.find(expected) > -1)


def get_api_key(filepath):
    """
    Retrieve the current OMDB API key, returns a string
    """

    with open(filepath, 'r') as api_file:
        return api_file.readline().replace('\n', '').replace('\r', '')


def imdb_titles(num_pages=1, content_type=['tv_series', 'mini_series'],
                genre_detail=None):
    """
    Scrapes imdb titles and IDs from top titles page
    returns list of dictionaries
    """
    titles = []

    content_search = ','.join(content_type)

    try:
        genre_search = ','.join(genre_detail)

    except TypeError:
        genre_search = None

    for i in range(1, num_pages+1):
        search_param = {
            'title_type': content_search,
            'page': str(i),
            'genres': genre_search
        }

        html_result = simple_get(
            'http://www.imdb.com/search/title', payload=search_param)

        soup = BeautifulSoup(html_result, 'html.parser')

        for struct in soup.findAll('div', {'class': 'lister-item-content'}):
            dict_builder = {}

            try:
                imdb_content = struct.find(
                    'span', {'class': 'userRatingValue'})
                dict_builder['imdb_id'] = imdb_content['data-tconst']

                title_content = struct.find('a')
                dict_builder['title'] = title_content.text

                rank_content = struct.find('span', {
                    'class': 'lister-item-index unbold text-primary'})
                dict_builder['rank'] = int(rank_content.text.replace('.', ''))

                genre_content = struct.find('span', {'class': 'genre'})
                dict_builder['genre'] = genre_content.text.replace(
                    ' ', '').replace('\n', '')

                rating_content = struct.find('div', {
                    'class': 'inline-block ratings-imdb-rating'})
                dict_builder['rating'] = rating_content['data-value']

            except TypeError:
                pass

            titles.append(dict_builder)

    return titles


def get_omdb_data(imdb_id=None, title=None, content_type=None, plot='short'):
    """
    Retrieve data from OMDB given parameter set, returns a JSON file
    """
    url = 'http://www.omdbapi.com/'
    payload = {
        'apikey': get_api_key('omdb_api_key.txt'),
        'type': content_type,
        'plot': plot
    }

    if imdb_id is None:
        if title is None:
            raise ValueError('Either a title or imdb_id must be specified')
        else:
            payload['t'] = title.replace(' ', '+')
    else:
        payload['i'] = imdb_id

    resp = simple_get(url, 'json', payload)
    json_resp = json.loads(resp)

    json_resp['Genre'] = json_resp['Genre'].replace(' ', '').split(',')

    return json_resp


OMDB_DATA = []

for d in imdb_titles(1, content_type=['movies']):
    try:
        result = get_omdb_data(title=d['title'])
        result['rank'] = d['rank']
        OMDB_DATA.append(result)
    except KeyError:
        try:
            result = get_omdb_data(imdb_id=d['imdb_id'])
            result['rank'] = d['rank']
            OMDB_DATA.append(result)

        except KeyError:
            pass

# with open('../data/test_output.txt', 'w') as f:
#     for line in OMDB_DATA:
#         f.write(str(line))

for content in OMDB_DATA:
    image_get(url=content['Poster'], filename=content['imdbID'])
