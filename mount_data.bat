@echo on
docker build -t posters:1.0 .
docker run -v C:/Users/Nick/code/poster-classifier/data:/home/data posters:1.0