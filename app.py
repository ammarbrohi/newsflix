from flask import Flask,request, url_for, redirect, render_template
import pickle
import os
import re
from tensorflow.python.keras.backend import set_session
import numpy as np
import pandas as pd
import tensorflow as tf 
from keras import backend
from keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
app = Flask(__name__)

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 25000
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
def auc(y_true, y_pred):
	auc = tf.metrics.auc(y_true, y_pred)[1]
	keras.backend.get_session().run(tf.local_variables_initializer())
	return auc

global graph

@app.route('/')
def hello_world():
	return render_template("index.html")
	
	
@app.route('/predict',methods=['POST'])
def predict():
	temp = pd.DataFrame([request.form['first_name']])
	temp = temp.iloc[0]
	temp = temp.apply(lambda x:re.sub('[!@#$:).;,?&]',' ',x.lower()))
	temp = temp.apply(lambda x:re.sub(' ', ' ', x))
	temp = tokenizer.texts_to_sequences(temp)
	temp = pad_sequences(temp,maxlen = MAX_SEQUENCE_LENGTH, padding ='pre', truncating ='pre')
	global model
	model = load_model('model.h5', custom_objects={'auc': auc})
	model._make_predict_function()
	temp = model.predict_on_batch(temp)
	print(temp[0])
	threshold = 0.5 #Threshold value here.
	if temp[0][0] < 0.5:
		return render_template('index.html', pred='Real')
	else:
		return render_template('index.html', pred='Fake')
		

if __name__ == '__main__':
		port = int(os.environ.get('PORT',5000))
		app.run(host='0.0.0.0', port=port,debug=True)
