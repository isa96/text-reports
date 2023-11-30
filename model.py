import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from keras import backend as K

import pickle

import text_preprocessing

class Att(Layer):
    def __init__(self, **kwargs):
        super(Att,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(Att, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        return K.sum(context, axis=1)
      
def load_model():
    dependencies = {'Att': Att()}
    model = tf.keras.models.load_model(
        'models/best_model.h5',
        custom_objects=dependencies
    )
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return [model, tokenizer]
  
def predict(model, text):
    # model[0] = model machine learning
    # model[1] = tokenizer
    labels_list=['gangguan ketenteraman dan ketertiban', 'jalan', 'jaringan listrik',
                 'parkir liar', 'pelayanan perhubungan', 'pohon',
                 'saluran air, kali/sungai', 'sampah', 'tata ruang dan bangunan',
                 'transportasi publik']
    dict_classes = dict(zip(range(len(labels_list)),
                            labels_list))
    
    # text preprocessing, include tokenizing
    preprocess_text = text_preprocessing.preprocess(text, stem=True)  # text_preprocessing
    text = model[1].texts_to_sequences([preprocess_text])   # tokenizer
    text = pad_sequences(text, padding='post',
                         maxlen=100, truncating='post')
    
    # predict the text
    prediction = model[0].predict(text, verbose=0)
    classes = np.argmax(prediction, axis = 1)
    pred_class = dict_classes[classes[0]]
    
    # create a table of predicted categories
    df = pd.Series(prediction[0].round(decimals=5) * 100, 
                   index=dict_classes.values()).sort_values(ascending=False)
    df = df.to_frame().reset_index()
    df = df.rename(columns={0: 'probability',
                            'index': 'prediksi_kategori_laporan'})
    
    return df, preprocess_text
