# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import codecs
import re
import os
import unicodedata

import numpy as np
import tensorflow as tf

from hyperparams import Hyperparams as hp
from hyperparams import lang2vocab

def load_vocab(lang="es"):
    vocab = lang2vocab(lang)

    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def text_normalize(text, lang="es"):
    vocab = lang2vocab(lang)

    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode="train", lang="es"):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab(lang)

    if mode=="train":
        if True:
            # Parse
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'transcript.csv')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            for line in lines:
                fname, _, text = line.strip().split("|")

                fpath = os.path.join(hp.data, "wavs", fname)
                fpaths.append(fpath)

                text = text_normalize(text, lang) + "E"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())

            return fpaths, text_lengths, texts
        else: # nick or kate
            # Parse
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'transcript.csv')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            for line in lines:
                fname, _, text, is_inside_quotes, duration = line.strip().split("|")
                duration = float(duration)
                if duration > 10. : continue

                fpath = os.path.join(hp.data, fname)
                fpaths.append(fpath)

                text += "E"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())

        return fpaths, text_lengths, texts

    else: # synthesize on unseen test text.
        # Parse
        lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:]
        sents = [text_normalize(line.split(" ", 1)[-1], lang).strip() + "E" for line in lines] # text normalization, E: EOS
        texts = np.zeros((len(sents), hp.max_N), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts

def load_text(text, lang="es"):
    char2idx, idx2char = load_vocab(lang)
    lines = text
    sents = [text_normalize(line, lang).strip() + "E" for line in lines] # text normalization, E: EOS
    texts = np.zeros((len(sents), hp.max_N), np.int32)
    for i, sent in enumerate(sents):
        texts[i, :len(sent)] = [char2idx[char] for char in sent]
    return texts