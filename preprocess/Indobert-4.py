#!/usr/bin/env python
# coding: utf-8

# # indolem/indobertweet-base-uncased

# In[1]:


#https://huggingface.co/indobenchmark/indobert-base-lite-p2
get_ipython().system('pip install transformers')


# In[2]:


gpu_info = get_ipython().getoutput('nvidia-smi')
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)


# In[3]:


get_ipython().run_line_magic('pip', 'install ntlk')


# In[4]:


get_ipython().run_line_magic('pip', 'install PySastrawi')


# In[5]:


import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[6]:


get_ipython().system('pip install datasets')


# In[7]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModel,BertTokenizer,AutoConfig
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline
# from datasets import load_metric
# from datasets import load_dataset
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


# In[8]:
df = pd.read_csv('/kaggle/input/product-review/PRDECT-ID.csv')
df.set_index(["Emotion","Category","Customer Review"]).value_counts('Emotion')


# In[9]:
kamus_alay = pd.read_csv('/kaggle/input/product-review/kamusalay.csv', encoding='ISO-8859-1', header = None)
kamus_alay_dict = kamus_alay.set_index(0).to_dict('dict')[1]
kamus_alay_dict


# In[10]:
df=df[['Customer Review','Emotion']]
df=df.rename(columns={"Customer Review": "text", "Emotion": "label"})
df.head()


# In[11]:


import string
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# In[12]:


def process_cleaning(text):

    # remove emoji spesifik, angka, url dulu
    text_cleaning_re = r"rt|url|[^\w\s]|'|nbsp|https\S+|[0-9]+"
    text_sub = re.sub(text_cleaning_re, ' ', str(text))

    # Remove strip / trims
    text_strip = text_sub.strip()

    # remove punctutation / tanda baca
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


# In[13]:


import numpy as np
import string

# data['cleaned without dict'] = data['Customer Review'].apply(process_cleaning2)
df['text'] = df['text'].apply(process_cleaning)
# print(list(data["processed_cleaning"]))

# x_cleaned = data['cleaned']
x_cleaned = df['text']
# x_cleaned2 = data['cleaned without dict']
# x_cleaned = data["cleaned"].values
df.head(10)


# In[14]:


from sklearn.preprocessing import LabelEncoder
from collections import Counter


# In[15]:


#encode label
print(Counter(df['label']))
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
# # print(data["emotion"].value_counts())
print(Counter(df['label']))
y_replaced = df['label']
print(y_replaced)


# In[16]:


df.head(5)


# In[17]:


from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x_cleaned, y_replaced, test_size=0.2, random_state=225)


# In[18]:


train_valid_ratio = 0.80
train_test_ratio = 0.20

#train test split
df_train, df_test = train_test_split(df, test_size = train_test_ratio, random_state = 42)
print("Train Shape",df_train.shape)
print("Test Shape",df_test.shape)
print(df_train)

# In[20]:

print(df.head())

print(df['label'])
pd.concat


# In[21]:


#prepare data train and validation
import pandas as pd
df_val = pd.DataFrame()



print(df_train.shape)
for row in df['label'].drop_duplicates():
  df_val = pd.concat([df_val, df_train.loc[df_train['label']==row]], ignore_index=True)

df_train = df_val
# print(df_val)
print(df_train.shape)

# In[22]:


print(df_train)


# In[23]:


#convert dataframe to dataset type
dataset_train = Dataset.from_dict(df_val)
dataset_val =  Dataset.from_dict(df_test)
print("Dataset Train : ",dataset_train)
print("Dataset Val : ",dataset_val)


# #Load Pre-trained Model and Fine Tune

# In[24]:


#konfigurasi penamaan label sesuai dengan kelas emoion yang digunakan
id2label = {
    "0": "Anger",
    "1": "Fear",
    "2": "Happy",
    "3": "Love",
    "4": "Sadness"
  }
label2id= {
    "Anger": 0,
    "Fear": 1,
    "Happy": 2,
    "Love": 3,
    "Sadness": 4
  }


# In[25]:
# https://huggingface.co/indobenchmark/indobert-lite-base-p2
# https://huggingface.co/ayameRushia/indobert-base-uncased-finetuned-indonlu-smsa
config = AutoConfig.from_pretrained("indolem/indobertweet-base-uncased")

config.label2id = label2id
config.id2label = id2label
config._num_labels = len(label2id)


# In[26]:
tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("indolem/indobertweet-base-uncased",config=config)
model.config


# In[27]:
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


# In[28]:
dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_val = dataset_val.map(tokenize_function, batched=True)


# In[29]:
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# In[30]:
training_args = TrainingArguments("test_trainer", eval_strategy="epoch",per_device_train_batch_size=8,num_train_epochs=8,learning_rate=2e-5,logging_steps=1, report_to="none")
trainer = Trainer(model=model.cuda(), args=training_args, train_dataset=dataset_train, eval_dataset=dataset_val,compute_metrics=compute_metrics)
trainer.train()


# In[31]:
model.save_pretrained("model")
tokenizer.save_pretrained("model")


# In[32]:
tokenizer = AutoTokenizer.from_pretrained("/kaggle/working/indobert-base-uncased-model")
model = AutoModelForSequenceClassification.from_pretrained("/kaggle/working/indobert-base-uncased-model")
model.eval()


# In[33]:
trainer.save_model('/kaggle/working/indobert-base-save_model')

# In[34]:
trainer.evaluate()
