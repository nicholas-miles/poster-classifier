"""
URL and JSON tools for OMDB retrieval
BeautifulSoup for web scraping
pandas for data analysis
"""
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import json
import pandas as df


def simple_get(url, expected='html', payload=None):
    """
    Attempts to get the content at \\url\\ by making an HTTP GET request
    """
    try:
        with closing(get(url, stream=True, params=payload)) as resp:
            if is_good_response(resp, expected):
                return resp.text
            else:
                return None

    except RequestException as e:
        print('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp, expected='html'):
    """
    Returns true if the response is the expected format, false otherwise
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find(expected) > -1)


def get_api_key(filepath):
    """
    Retrieve the current OMDB API key, returns a string
    """

    with open(filepath, 'r') as api_file:
        return api_file.readline().replace('\n', '').replace('\r', '')


def imdb_titles(num_pages=1):
    """
    Scrapes imdb titles and IDs from top titles page
    returns list of dictionaries
    """
    titles = []

    for i in range(num_pages):
        search_param = {'title_type': 'tv_series,mini_series', 'page': str(i)}
        html_result = simple_get('http://www.imdb.com/search/title', payload=search_param)

        soup = BeautifulSoup(html_result, 'html.parser')

        for struct in soup.findAll('div', {'class': 'lister-item-content'}):
            d = {}

            try:
                d['title'] = struct.find('a').text
                d['genre'] = struct.find('span', {'class': 'genre'}).text.replace(' ','').replace('\n', '')
                d['rating'] = struct.find('div', {'class': 'inline-block ratings-imdb-rating'})['data-value']
                d['imdb_id'] = struct.find('span', {'class': 'userRatingValue'})['data-tconst']

            except TypeError:
                pass

            titles.append(d)

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
    return json.loads(resp)


for d in imdb_titles(3):
    print(d['title'])

#with open('omdb_test.json', 'w') as outfile:
#    json.dump(get_omdb_data(title='Game of Thrones'), outfile)

#search_string = 'http://www.omdbapi.com/?apikey=' + get_api_key('omdb_api_key.txt')
#
#a = get_omdb_data(title='Game of Thrones')
# print(a)

#TEST_TITLES = ['The Office', 'Parks and Recreation', 'Game of Thrones']

#RAW_DATA = []

# for T in TEST_TITLES:
#     RAW_DATA.append(get_omdb_data(title=T, plot='full'))
#
# DATA = df.DataFrame(RAW_DATA)
