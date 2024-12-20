# %% [markdown]
# # indolem/indobertweet-base-uncased

# %%
#https://huggingface.co/indobenchmark/indobert-base-lite-p2
# %%
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

# %%
#Loading the data train .txt file
# df = pd.read_csv("/content/gdrive/MyDrive/ColabNotebooks/Chatbot/Experimen 3/Data/Dataset/Train/Dataset Tokopedia Review - Train (Semicolon Delimited).txt" , sep=';',encoding= 'unicode_escape')
# df = pd.read_csv("sample_data/PRDECT-ID.csv" , encoding='ISO-8859-1')
df = pd.read_csv('/kaggle/input/product-review/PRDECT-ID.csv')
# df.set_index(["labelEmotions","category","reviewText"]).count(level="labelEmotions")
df.set_index(["Emotion","Category","Customer Review"]).value_counts('Emotion')

# %%
# kamus_alay = pd.read_csv("sample_data/kamusalay.csv" , encoding='ISO-8859-1',header = None)
kamus_alay = pd.read_csv('/kaggle/input/product-review/kamusalay.csv', encoding='ISO-8859-1', header = None)
kamus_alay_dict = kamus_alay.set_index(0).to_dict('dict')[1]
kamus_alay_dict

# %%
#rename column
# df=df[['reviewText','labelEmotions']]
# df=df.rename(columns={"reviewText": "text", "labelEmotions": "label"})

df=df[['Customer Review','Emotion']]
df=df.rename(columns={"Customer Review": "text", "Emotion": "label"})
df.head()

# %%
import string
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# %%
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

# %%
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

# %%
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# %%
#encode label
# def label2id (row):
#   if row['label'] == "Sadness":
#     return 0
#   if row['label'] == "Anger" :
#     return 1
#   if row['label'] == "Love" :
#     return 2
#   if row['label'] == "Fear":
#     return 3
#   if row['label'] == "Happy":
#     return 4

print(Counter(df['label']))
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
# # print(data["emotion"].value_counts())
print(Counter(df['label']))
y_replaced = df['label']
print(y_replaced)

# %%
df.head(5)

# %%
from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x_cleaned, y_replaced, test_size=0.2, random_state=225)

# %%
train_valid_ratio = 0.80
train_test_ratio = 0.20

#train test split
df_train, df_test = train_test_split(df, test_size = train_test_ratio, random_state = 42)
print("Train Shape",df_train.shape)
print("Test Shape",df_test.shape)

print(df_train)
#train valid split
# df_train, df_valid = train_test_split(df_train, train_size = train_valid_ratio, random_state = 42)
# print("Train Shape",df_train.shape)
# print("Valid Shape",df_valid.shape)

# %%
# df['labelEncoded'] = df.apply(lambda row: label2id(row), axis=1)
# df=df.rename(columns={"label": "emotions", "labelEncoded": "label"})
# print(df.head())
# label_reference=df[['emotions','label']].copy().drop_duplicates()
# print(label_reference)

# %%
# df=df.rename(columns={"label": "emotions", "labelEncoded": "label"})
print(df.head())
# label_reference=df[['emotions','label']].copy().drop_duplicates()
# print(df[['emotions','label']].copy().drop_duplicates())
# print(df['Emotion'].copy().drop_duplicates())
# print(df['label'].copy().drop_duplicates())
# print(df.rename(columns={'label': 'emotions', 'labelEncoded': 'label'}))
# print(df['label'].drop_duplicates())
print(df['label'])
pd.concat

# %%
#prepare data train and validation
import pandas as pd
df_val = pd.DataFrame()


# pd.concat(df, ignore_index=True)
# print(range(df['label'].drop_duplicates()))
# df[df['label']]
# print(df[df['label']] == )
print(df_train.shape)
for row in df['label'].drop_duplicates():
  df_val = pd.concat([df_val, df_train.loc[df_train['label']==row]], ignore_index=True)
  # print(df.loc[df['label']==row])
# lanjut develop di sini
# df_train = df[~df.text.isin(df_val.text)].copy()
df_train = df_val
# print(df_val)
print(df_train.shape)
# df_train = df[~df.text.isin(df_val.text)].copy()
# df_train.shape
# print(df_train)
# df_train = df[~df.text.isin(df_val.text)].copy()
# print(df_train.shape)
# # df_train = pd.concat([])
# df_train
# for index,sentence in df[['label']].drop_duplicates().iterrows():
#   # df_val = pd.concat([df_val,df[df.loc['label']==sentence.item()].head(5)])
#   df_val=df_val.append(df[df['label']==sentence.item()].head(5))
#   # df_val=df_val.append(df.loc[df['label']==sentence.item()].head(5))
# df_train = df[~df.text.isin(df_val.text)].copy()

# %%
print(df_train)

# %%
#convert dataframe to dataset type
dataset_train = Dataset.from_dict(df_val)
dataset_val =  Dataset.from_dict(df_test)
print("Dataset Train : ",dataset_train)
print("Dataset Val : ",dataset_val)

# %% [markdown]
# #Load Pre-trained Model and Fine Tune

# %%
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

# %%
# https://huggingface.co/indobenchmark/indobert-lite-base-p2
# https://huggingface.co/ayameRushia/indobert-base-uncased-finetuned-indonlu-smsa
config = AutoConfig.from_pretrained("indolem/indobertweet-base-uncased")

config.label2id = label2id
config.id2label = id2label
config._num_labels = len(label2id)

# %%
tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("indolem/indobertweet-base-uncased",config=config)
model.config

# %%
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# %%
dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_val = dataset_val.map(tokenize_function, batched=True)

# %%
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

# %%
training_args = TrainingArguments("test_trainer", eval_strategy="epoch",per_device_train_batch_size=8,num_train_epochs=8,learning_rate=2e-5,logging_steps=1, report_to="none")
trainer = Trainer(model=model.cuda(), args=training_args, train_dataset=dataset_train, eval_dataset=dataset_val,compute_metrics=compute_metrics)
trainer.train()

# %%
model.save_pretrained("/model/deep_learning/save_pretrained/indobert-base-uncased-model")
tokenizer.save_pretrained("/model/deep_learning/save_pretrained/indobert-base-uncased-model")

# %%
tokenizer = AutoTokenizer.from_pretrained("/model/deep_learning/save_pretrained/indobert-base-uncased-model")
model = AutoModelForSequenceClassification.from_pretrained("/model/deep_learning/save_pretrained/indobert-base-uncased-model")
model.eval()

# %%
trainer.save_model('/model/deep_learning/save_model/indobert-base-uncased-model')

# %%
trainer.evaluate()

# %%
#convert dataframe to dataset type
dataset_test= Dataset.from_dict(df_test)
dataset_test = dataset_test.map(tokenize_function, batched=True)

# %%
predicted_review = trainer.predict(dataset_test)
raw_pred, _, _ = predicted_review
predclas= np.argmax(raw_pred, axis=1)

# %%
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# %%
labels = df_test["label"].unique()
accuracy = accuracy_score(df_test['label'], predclas)
print('Accuracy: ', accuracy * 100)
precision = precision_score(df_test['label'], predclas, average='macro', zero_division=1)
print('Precision: ', precision * 100)
# recall: tp / (tp + fn)
recall = recall_score(df_test['label'], predclas, average='macro', zero_division=1)
print('Recall: ', recall * 100)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(df_test['label'], predclas, average='macro')
print('F1 score: ', f1 * 100)
print('Classification Report:')
print(classification_report(df_test['label'], predclas, labels=labels, digits=4))
cm = confusion_matrix(df_test['label'], predclas, labels=labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

ax.set_title('Confusion Matrix')

ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')