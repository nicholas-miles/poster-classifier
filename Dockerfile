FROM python:3

RUN python -m pip install --upgrade pip && \
	pip install bs4 \
				matplotlib \
				numpy \
				opencv-python \
				pandas \
				seaborn \
				sklearn \
				tqdm

WORKDIR /home

CMD ["mkdir", "code"]

ADD /code/omdb_scraper.py /home/code/
ADD /code/image_analysis.py /home/code/
ADD /code/genre_clustering.py /home/code/

WORKDIR /home/code

ENV MPLBACKEND="agg"

CMD ["python","genre_clustering.py"]