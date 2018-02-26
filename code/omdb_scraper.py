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

def simple_get(url, expected='html'):
    """
    Attempts to get the content at \\url\\ by making an HTTP GET request
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp, expected):
                return resp.content
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


def get_omdb_data(imdb_id=None, title=None, content_type=None,
                  plot='short', return_type='json'):
                  
    """
    Retrieve data from OMDB given parameter set, returns a JSON file
    """
    
    search_string = "http://www.omdbapi.com/?apikey=" + get_api_key('omdb_api_key.txt')
    if imdb_id is None:
        if title is None:
            raise ValueError("Either a title or imdb_id must be specified")
        else:
            search_string += "&t=" + title.replace(' ', '+')
    else:
        search_string += "&i=" + imdb_id

    if content_type is not None:
        search_string += "&type=" + content_type

    search_string += "&plot=" + plot + "&r=" + return_type

    resp = simple_get(search_string, 'json')
    return json.loads(resp)

search_string = "http://www.omdbapi.com/?apikey=" + get_api_key('omdb_api_key.txt')

a = get_omdb_data(title='Game of Thrones')
print(a)

#TEST_TITLES = ['The Office', 'Parks and Recreation', 'Game of Thrones']

#RAW_DATA = []

# for T in TEST_TITLES:
#     RAW_DATA.append(get_omdb_data(title=T, plot='full'))
# 
# DATA = df.DataFrame(RAW_DATA)
