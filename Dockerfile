FROM python:3.7.3-slim-stretch

RUN apt-get -y update && apt-get -y install gcc && apt-get -y install libsndfile1-dev

WORKDIR /
COPY logdir /logdir

# Make changes to the requirements/app here.
# This Dockerfile order allows Docker to cache the checkpoint layer
# and improve build times if making changes.
RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install sndfile librosa TensorFlow==1.15.4 tqdm matplotlib scipy starlette uvicorn ujson aiofiles
COPY *.py /

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENTRYPOINT ["python3", "-X", "utf8", "app.py"]

