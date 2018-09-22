from __future__ import print_function
import numpy as np
from numpy import distutils
import pandas as pd

from keras.preprocessing import sequence
from keras.losses import categorical_crossentropy
from keras.models import model_from_json
from collections import OrderedDict
from nltk.corpus import stopwords
import _pickle as pkl
from keras import optimizers , utils
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge,Conv1D,MaxPooling1D , Activation
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_features = 5000
maxlen = 100

testPath = './testText.txt'

test = []
tt = open(testPath, 'r')

test = tt.readlines()


tokenizer=Tokenizer(nb_words=max_features)
tokenizer.fit_on_texts(test)
sequence=tokenizer.texts_to_sequences(test)
word_index=tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_test = pad_sequences(sequence, maxlen=maxlen)


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

classes = model.predict_classes(X_test)
datatest=pd.read_csv('test.csv')
datatest['Score']=classes
datatest.to_csv('outputMSE.csv',index=False)
