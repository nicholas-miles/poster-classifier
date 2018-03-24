import pickle
import os
import pandas as pd
from PIL import Image
from image_analysis import *

if __name__ == '__main__':
	km = pickle.load(open('../data/out/km.pickle', 'rb'))
	# str_dir = os.fsencode()
	files = [ '../data/posters/test/' + os.fsdecode(x) for x in os.listdir('../data/posters/test/') ]

	img = pd.DataFrame(load_images(files, None, None))

	for _, im in img.iterrows():
		fim_t = np.argmin(km.transform(im.flat_image), axis=1)
		centers = np.asarray([ km.cluster_centers_[x] for x in fim_t])
		centers = np.rint(centers).astype('uint8').reshape(im.image.shape)

		img = Image.fromarray(centers, 'RGB')
		img.show()
		img.save('../data/posters/test/' + im.imdb_id + '_TRANSFORM.jpg')

