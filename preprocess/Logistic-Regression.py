#!/usr/bin/env python
# coding: utf-8

# In[1]:

import nltk
nltk.download('punkt')


# In[2]:


# get_ipython().run_line_magic('pip', 'install pandas')


# In[3]:
# Import Library
import pandas as pd

# In[4]:
data = pd.read_csv('dataset/PRDECT-ID.csv')


# In[5]:
kamus_tb = pd.read_csv('dataset/kamusalay.csv', encoding='ISO-8859-1', header = None)
kamus_tb_dict = kamus_tb.set_index(0).to_dict('dict')[1]


# In[6]:
from collections import Counter
data = pd.DataFrame(data)
data = data[['Customer Review', 'Emotion']]


# In[7]:
import string
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# In[9]:
# use kamus alay

def process_cleaning(text):

    # Replace emoji, numerik, url, non-word character by spacing
    text_cleaning_re = r"rt|url|[^\w\s]|'|nbsp|https\S+|[0-9]+"
    text_sub = re.sub(text_cleaning_re, ' ', str(text))

    # Remove strip / trims
    text_strip = text_sub.strip()

    # Remove punctutation / tanda baca
    translator = str.maketrans('', '', string.punctuation)
    text_no_punct = text_strip.translate(translator)

    # Lower Case
    text_lower = text_no_punct.casefold()

    # Tokenize
    text_token_stan = word_tokenize(text_lower)

    # Penggantian kata tidak baku / Normalization
    word_dict = []
    for word in text_token_stan:
        word_dict.append(kamus_tb_dict.get(word, word))
    tokens = " ".join(word_dict)

    # Tambah kata singkatan 
    more_stopword = ["sih","nya"]

    # menampung stopword ke variabel untuk jadi operator remove stopword
    stopword_user = StopWordRemoverFactory().get_stop_words() + more_stopword

    # Remove stopword
    token_new = word_tokenize(tokens)
    filter_new = []
    filter_new2 = [word.strip() for word in token_new]
    filter_new = [word for word in filter_new2 if not word in stopword_user]
    tokens = " ".join(filter_new)

    return tokens



# In[10]:
import numpy as np
import string

data["cleaned"] = data["Customer Review"].apply(process_cleaning)
x_cleaned = data["cleaned"]
data.head(10)


# In[11]:
from sklearn.preprocessing import LabelEncoder
print(Counter(data["Emotion"]))
label_encoder = LabelEncoder()
data["emotion"] = label_encoder.fit_transform(data["Emotion"])
y_replaced = data["emotion"]


# In[12]:

from sklearn.linear_model import LogisticRegression
# Create a model
lr = LogisticRegression()

# In[13]:
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


# In[16]:
tf_vect = TfidfVectorizer()

x = x_cleaned
y = y_replaced # emotion field

# splitting X and y into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=225)

print('Shape of X Training Data :', x_train.shape)
print('Shape of Y Training Data :', y_train.shape)
print('Shape of X Testing Data : ', x_test.shape)
print('Shape of Y Testing Data : ', y_test.shape)
print('Length of X Training Data :', len(x_train))
print('Length of Y Training Data :', len(y_train))
print('Length of X Testing Data : ', len(x_test))
print('Length of Y Testing Data : ', len(y_test))

model_lr_tf = Pipeline([('vectorizer',tf_vect),('classifier',lr)])
# y_emot = y_train.replace({0: 'Anger', 1: 'Fear', 2: 'Happy', 3:'Love', 4: 'Sadness'})
lr = model_lr_tf.fit(x_train, y_train)
# In[17]:

import pickle
pickle.dump(lr, open('model/machine_learning/lr.pkl', 'wb'))


