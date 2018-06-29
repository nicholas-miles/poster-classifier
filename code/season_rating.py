from omdb_scraper import *
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def df_seasons(imdb_id):
	ratings = get_season_ratings(imdb_id)
	dfo = pd.DataFrame(columns=['rating','votes','season', 'episode'])
	for season in range(len(ratings)):
		df = pd.DataFrame(ratings[season])
		df['season'] = season + 1
		df['episode'] = df.index + 1
		dfo = dfo.append(df)

	return dfo.reset_index().drop(['index'], axis=1)


def get_season_ratings(imdb_id):
	url = simple_get('https://www.imdb.com/title/' + imdb_id)
	soup = BeautifulSoup(url, 'html.parser')

	seasons_and_years = soup.find('div', {'class': "seasons-and-year-nav"})\
						    .find_all('a')

	seasons = sorted([int(url.string) for url in seasons_and_years if str(url).find('season') != -1])
	raw_ratings = []
	final_ratings = []

	for num in seasons:
		payload = {'season' : num}
		url = simple_get('https://www.imdb.com/title/{}/episodes'.format(imdb_id), payload=payload)
		
		season_soup = BeautifulSoup(url, 'html.parser')
		all_ratings = season_soup.find_all('div', {'class': 'ipl-rating-star '})
		
		episode_ratings = []
		for stub in all_ratings:
			episode = {}
			episode_soup = BeautifulSoup(str(stub), 'html.parser')
			episode['rating'] = float(episode_soup
										.find('span', {'class': 'ipl-rating-star__rating'})
										.get_text())
			episode['votes'] = 	int(episode_soup
										.find('span', {'class': 'ipl-rating-star__total-votes'})
										.get_text()[1:-1]
										.replace(',', ''))


			episode_ratings.append(episode)

		final_ratings.append(episode_ratings)

	return final_ratings


if __name__ == '__main__':
	df = df_seasons('tt0386676')
	print(df.head())
	raise SystemExit
	
	CONTENT = 500
	TYPE = 'tv'

	OMDB_DATA = []

	for i in tqdm(range(1, CONTENT // 50 + 1)):
		search_param = {'title_type': TYPE, 'page': str(i)}

		for val in tqdm(imdb_titles(search_param)):
			OMDB_DATA.append(get_omdb_data(val))

	print(OMDB_DATA[0])