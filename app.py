from flask import Flask, render_template as rt, request, url_for
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

# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# DEEP LEARNING
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# # path = r"preprocess\indobertweet-base-uncased-model"
# path_save_pretrained = r"model\deep_learning\save_pretrained_old"
# tokenizer = AutoTokenizer.from_pretrained(path_save_pretrained, local_files_only=True)# load tokenizer
# model = AutoModelForSequenceClassification.from_pretrained(path_save_pretrained, local_files_only=True)
# @app.route('/api/result', methods=['POST'])
# def result():    
    
#     # Get the sample text from the POST request.
#     review = request.form['review']

#     # Tokenize the input text review
#     inputs = tokenizer(review, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Get the predicted class
    # emotion = torch.argmax(outputs.logits, dim=-1).item()
#     emotion = torch.argmax(outputs.logits, axis=-1).item()
#     if emotion == 0 : 
#         emosiLabel = 'Marah' 
#         emotion = 'Anger'
#         path_img = url_for('static', filename='image/Anger.png')
#     elif emotion == 1 : 
#         emosiLabel = 'Takut'
#         emotion = 'Fear'
#         path_img = url_for('static', filename='image/Fear.png')
#     elif emotion == 4 : 
#         emosiLabel = 'Kecewa'
#         emotion = 'Sadness'
#         path_img = url_for('static', filename='image/Sadness.png')
#     elif emotion == 2 : 
#         emosiLabel = 'Senang'
#         emotion = 'Happy'
#         path_img = url_for('static', filename='image/Happy.png')
#     elif emotion == 3 : 
#         emosiLabel = 'Suka'
#         emotion = 'Love'
#         path_img = url_for('static', filename='image/Love.png')
#     return rt('main.html', emotion=emotion, emosi=emosiLabel, img_emotion=path_img, input=review)

# MACHINE LEARNING
import pickle
# syntax:
# alternatif 1
# pickle.dump(namamodel, open("model.pkl", "rb"))
# alternatif 2
# with open('model/svc.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

    # pickle.dump(pipeline, file)
model = pickle.load(open("model/machine_learning/lr_old/lr.pkl", "rb"))

@app.route('/api/result', methods=['POST'])
def result():    
    review = request.form['review']
    emotion = model.predict([review])
    emotion = str(emotion).strip("[]").replace("'", "")
    if emotion == '0' : 
        emosiLabel = 'Marah' 
        emotion = 'Anger'
        path_img = url_for('static', filename='image/Anger.png')
    elif emotion == '1' : 
        emosiLabel = 'Takut'
        emotion = 'Fear'
        path_img = url_for('static', filename='image/Fear.png')
    elif emotion == '4' : 
        emosiLabel = 'Kecewa'
        emotion = 'Sadness'
        path_img = url_for('static', filename='image/Sadness.png')
    elif emotion == '2' : 
        emosiLabel = 'Senang'
        emotion = 'Happy'
        path_img = url_for('static', filename='image/Happy.png')
    elif emotion == '3' : 
        emosiLabel = 'Suka'
        emotion = 'Love'
        path_img = url_for('static', filename='image/Love.png')
    return rt('main.html', emotion=emotion, emosi=emosiLabel, img_emotion=path_img, input=review)

# main driver function
if __name__ == '__main__':
    # app.run(debug = True, host= '192.168.8.100')
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(host="0.0.0.0", port=5000, debug = True)
    

