# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function
import os
import gc
import logging

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from scipy.io.wavfile import write
from tqdm import tqdm
from starlette.applications import Starlette
from starlette.responses import FileResponse
import uvicorn
from pydub import AudioSegment

from utils import spectrogram2wav
from hyperparams import Hyperparams as hp
from graph import Graph
from data_load import load_text

lang = {"cortazar":"es", "gibi":"en", "kid":"en", "spinetta":"es", "rapbot": "es"}

def clean_text(text):
	return text.replace("ñ","ni") + "      "

def synthesize_full(model_name, texts):
	tf.reset_default_graph()
	g = Graph(lang=lang[model_name])
	texts = [clean_text(text) for text in texts]
	print("Graph loaded")
	with tf.Session() as sess:

	    sess.run(tf.global_variables_initializer())

	    # Restore parameters
	    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
	    saver1 = tf.train.Saver(var_list=var_list)
	    saver1.restore(sess, tf.train.latest_checkpoint(os.path.join("models",model_name, "logdir-1")))
	    print("Text2Mel Restored!")

	    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
	               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
	    saver2 = tf.train.Saver(var_list=var_list)
	    saver2.restore(sess, tf.train.latest_checkpoint(os.path.join("models",model_name, "logdir-2")))
	    print("SSRN Restored!")

	    if len(texts) > 0:
	        L = load_text(texts,lang[model_name])
	        #print(L)
	        max_T = min(int(sum([len(text) for text in texts])*1.5), hp.max_T)
	        # Feed Forward
	        ## mel
	        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
	        prev_max_attentions = np.zeros((len(L),), np.int32)
	        for j in tqdm(range(max_T)):
	            _gs, _Y, _max_attentions, _alignments = \
	                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
	                         {g.L: L,
	                          g.mels: Y,
	                          g.prev_max_attentions: prev_max_attentions})
	            Y[:, j, :] = _Y[:, j, :]
	            prev_max_attentions = _max_attentions[:, j]

	        # Get magnitude
	        Z = sess.run(g.Z, {g.Y: Y})

	        for i, mag in enumerate(Z):
	            print("Working on file", i+1)
	            wav = spectrogram2wav(mag)
	            write(f"{i}.wav", hp.sr, wav)
	            break

def wav2mp3(path_to_file):
	final_audio = AudioSegment.from_wav(file=path_to_file)
	path_to_file = path_to_file.replace(".wav",".mp3")
	final_audio.export(path_to_file, format="mp3")
	return path_to_file	

app = Starlette(debug=False)

# Needed to avoid cross-domain issues
response_header = {
    'Access-Control-Allow-Origin': '*',
}

@app.route('/', methods=['GET', 'POST'])
async def homepage(request):
	if request.method == 'GET':
	    params = request.query_params
	elif request.method == 'POST':
	    params = await request.json()

	text = params.get('text', 'Hola')
	model = params.get('model', 'rapbot')
	synthesize_full(model, [text])
	path_to_file = "0.wav"
	gc.collect()
	return FileResponse(wav2mp3(path_to_file), headers=response_header)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
