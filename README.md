# A TensorFlow Implementation of DC-TTS: yet another text-to-speech model

## Pretrained Model for LJ

Download [this](https://www.dropbox.com/s/1oyipstjxh2n5wo/LJ_logdir.tar?dl=0).

## Installation

```bash
docker build . -t tts


docker run -p 8080:8080 --memory="2g" --cpus="1" tts


docker tag gpt2 gcr.io/[PROJECT-ID]/tts
docker push gcr.io/[PROJECT-ID]/tts

```
