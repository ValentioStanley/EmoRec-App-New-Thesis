#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('pip', 'install emoji')
# get_ipython().run_line_magic('pip', 'install PySastrawi')
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


kamus_alay = pd.read_csv('dataset/kamusalay.csv', encoding='ISO-8859-1', header = None)
kamus_alay_dict = kamus_alay.set_index(0).to_dict('dict')[1]


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


# In[8]:


from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# for lol in StopWordRemoverFactory().get_stop_words():
#   print(lol)


# In[9]:


# pake kamus alay

def process_cleaning(text):

    # remove emoji spesifik, angka, url dulu
    text_cleaning_re = r"rt|url|[^\w\s]|'|nbsp|https\S+|[0-9]+"
    text_sub = re.sub(text_cleaning_re, ' ', str(text))

    # Remove strip / trims
    text_strip = text_sub.strip()

    # remove punctutation/simbolll
    translator = str.maketrans('', '', string.punctuation)
    text_no_punct = text_strip.translate(translator)

    # Lower Case
    text_lower = text_no_punct.casefold()

    # tokenize
    text_token_stan = word_tokenize(text_lower)

    word_dict = []
    for word in text_token_stan:
        word_dict.append(kamus_alay_dict.get(word, word))
    tokens = " ".join(word_dict)

    # tambah kata singkatan =
    more_stopword = ["sih","nya"]

    # menampung stopword ke variabel untuk jadi operator remove stopword
    stopword_user = StopWordRemoverFactory().get_stop_words() + more_stopword
    # stopword_user = StopWordRemoverFactory().get_stop_words()

    # alternatif untuk perbaiki akurasi
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
# print(np.array(data["processed_cleaning"]))
# print(list(data["processed_cleaning"]))

x_cleaned = data["cleaned"]
# x_cleaned = data["cleaned"].values
data.head(10)


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


print(Counter(data["Emotion"]))
label_encoder = LabelEncoder()
data["emotion"] = label_encoder.fit_transform(data["Emotion"])
y_replaced = data["emotion"]


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


# Create a model
lr = LogisticRegression()


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


# In[16]:


co_vect = CountVectorizer()
tf_vect = TfidfVectorizer()
hash_vect = HashingVectorizer()

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


