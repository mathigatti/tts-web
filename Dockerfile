FROM python:3.7.3-slim-stretch

RUN apt-get -y update && apt-get -y install gcc && apt-get -y install libsndfile1-dev && apt-get -y install ffmpeg

WORKDIR /
COPY models /models

# Make changes to the requirements/app here.
# This Dockerfile order allows Docker to cache the checkpoint layer
# and improve build times if making changes.
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip --no-cache-dir install librosa TensorFlow==1.15.4 pydub tqdm scipy starlette uvicorn ujson aiofiles
COPY *.py /

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENTRYPOINT ["python3", "-X", "utf8", "app.py"]

