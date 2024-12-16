from flask import Flask, render_template as rt, request
# from model.Implementation_Testing import process_cleaning

# Flask constructor takes the name of 
# current module (__name__) as argument.
# nama aplikasi/title app dengan flask
app = Flask(__name__)

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
# rt = render_template
@app.route('/')
def index():
    return rt('main.html')

# DEEP LEARNING
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
path = r"preprocess\indobertweet-base-uncased-model"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)# load tokenizer
# D:\Projects\Product-Review-Data-Mining-App-Thesis\preprocess\indobertweet-base-uncased-model
model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
@app.route('/api/result', methods=['POST'])
def result():    
    
    # Get the sample text from the POST request.
    review = request.form['review']

    # Tokenize the input text review
    inputs = tokenizer(review, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class
    emotion = torch.argmax(outputs.logits, dim=-1).item()
    if emotion == 0 : emosiLabel = 'Marah'
    elif emotion == 1 : emosiLabel = 'Takut'
    elif emotion == 4 : emosiLabel = 'Kecewa'
    elif emotion == 2 : emosiLabel = 'Senang'
    elif emotion == 3 : emosiLabel = 'Suka'
    # print(f"Predicted class: {predictions.item()}")
    return rt('main.html', emotion=str(emosiLabel), emosi=emosiLabel, input=review)

# MACHINE LEARNING
import pickle
# syntax:
# alternatif 1
# pickle.dump(namamodel, open("model.pkl", "rb"))
# alternatif 2
# with open('model/svc.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

#     pickle.dump(pipeline, file)
# model = pickle.load(open("model/lr.pkl", "rb"))
# tfid_vect = pickle.load(open("model/tf_vect.pkl", "rb"))

# @app.route('/api/result', methods=['POST'])
# def result():    
#     review = request.form['review']
#     emotion = model.predict([review])
#     emotion = str(emotion).strip("[]").replace("'", "")
#     if emotion == 'Anger': emosiLabel = 'Marah'
#     elif emotion == 'Fear' : emosiLabel = 'Takut'
#     elif emotion == 'Sadness' : emosiLabel = 'Kecewa'
#     elif emotion == 'Happy' : emosiLabel = 'Senang'
#     elif emotion == 'Love' : emosiLabel = 'Suka'
#     return rt('main.html', emotion=emotion, emosi=emosiLabel, input=review)

    # Extract input features from the request
    # review = float(request.form['review'])
    # print(df_train.shape)
    # for row in review.drop_duplicates():
        # df_val = pd.concat([df_val, review.loc[review==row]], ignore_index=True)
        
        
    # emotion = float(request.form['emotion'])
    # Preprocess the input features
    # Make predictions using the loaded model
    # form = np.array([[review, emotion]]).reshape(1, 12) 
    # prediction = model.predict(form)   

# main driver function
if __name__ == '__main__':
    # app.run(debug = True, host= '192.168.8.100')
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug = True)
    

