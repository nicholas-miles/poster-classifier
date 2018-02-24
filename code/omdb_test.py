"""importing url and JSON tools for OMDB retrieval, and pandas for data analysis"""
from urllib.request import urlopen
import json
import pandas as df

def get_api_key(filepath):
    """Retrieve the current OMDB API key, returns a string"""
    with open(filepath, 'r') as api_file:
        return api_file.readline().replace('\n', '').replace('\r', '')

def get_omdb_data(imdb_id=None, title=None, content_type=None, plot='short', return_type='json'):
    """Retrieve data from OMDB given parameter set, returns a JSON file"""
    search_string = "http://www.omdbapi.com/?apikey=" + get_api_key('omdb_api_key.txt')
    if imdb_id is None:
        if title is None:
            raise ValueError("Either a title (t) or ID (i) must be specified")
        else:
            search_string += "&t=" + title.replace(' ', '+')
    else:
        search_string += "&i=" + imdb_id

    if content_type is not None:
        search_string += "&type=" + content_type

    search_string += "&plot=" + plot + "&r=" + return_type

    with urlopen(search_string) as url:
        return json.loads(url.read().decode())


TEST_TITLES = ['The Office', 'Parks and Recreation', 'Game of Thrones']

RAW_DATA = []

for T in TEST_TITLES:
    RAW_DATA.append(get_omdb_data(title=T, plot='full'))

DATA = df.DataFrame(RAW_DATA)
