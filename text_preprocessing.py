import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import pandas as pd

alay_df = pd.read_csv('data/new_kamusalay.csv', 
                      encoding = 'latin-1', 
                      header = None)

alay_df.rename(columns={0: 'original', 
                        1: 'replacement'},
               inplace = True)

alay_dict_map = dict(zip(alay_df['original'], alay_df['replacement']))

new_alay = dict()

def normalize_alay(text):
  global new_alay
  for word in text.split():
    if word in alay_dict_map:
      if word not in new_alay:
        new_alay[word] = alay_dict_map[word]
  return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

def convert_lower_case(text):
  return text.lower()

def remove_stop_words(text):
  stop_words = stopwords.words('indonesian')
  words = word_tokenize(str(text))
  new_text = ""
  for w in words:
    if w not in stop_words and len(w) > 1:
      new_text = new_text + " " + w
  return new_text

def remove_unnecessary_char(text):
  text = re.sub('permasalahan:\n',' ',text) # Remove every 'Permasalahan:\n'
  text = re.sub('  +', ' ', text) # Remove extra spaces
  text = re.sub(' jam ', ' ', text) # Remove tulisan "jam"
  text = re.sub(' pagi ', ' ', text) # Remove tulisan "pagi"
  text = re.sub(' sore ', ' ', text) # Remove tulisan "sore"
  i_text = text.find('lokasi:')
  if i_text != -1:
    text = text[:i_text]
  return text

def remove_punctuation(text):
  symbols = string.punctuation
  for i in range(len(symbols)):
    text = text.replace(symbols[i], ' ')
    text = text.replace("  ", " ")
  text = text.replace(',', '')
  return text

def stemming(text):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  
  tokens = word_tokenize(str(text))
  new_text = ""
  for w in tokens:
    new_text = new_text + " " + stemmer.stem(w)
  return new_text

def preprocess(text, stem=False):
  global counter

  text = convert_lower_case(text)
  text = remove_unnecessary_char(text)
  text = remove_punctuation(text)
  text = normalize_alay(text)
  text = remove_stop_words(text)
  if stem == True:
    text = stemming(text)
  text = text.strip()

  return text
