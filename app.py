
from flask import Flask,render_template,request
import google.generativeai as genai
import os
import numpy as np
import textblob


import joblib
from transformers import BertTokenizer, BertModel
import torch

# Load the BERT model classifier
bert_model = joblib.load('bert_model.pkl')

# Load the BERT tokenizer (ensure it's the same tokenizer used in training)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the BERT model (if you need embeddings, but the classifier is separate)
bert_embedding_model = BertModel.from_pretrained('bert-base-uncased')



api = os.getenv("MAKERSUITE_API_TOKEN")
model = genai.GenerativeModel("gemini-1.5-flash")
genai.configure(api_key="AIzaSyCykAjoIpT0jrqQW1R1ExZ4G404ngOMky4")

app = Flask(__name__)
user_name = ""
flag = 1

@app.route("/",methods=["GET","POST"])
def index():
    global flag
    flag = 1
    return(render_template("index.html"))

@app.route("/main",methods=["GET","POST"])
def main():
    global flag,user_name
    if flag==1:
        user_name = request.form.get("q")
        flag = 0
    return(render_template("main.html",r=user_name))

@app.route("/prediction",methods=["GET","POST"])
def prediction():
    return(render_template("prediction.html"))

@app.route("/DBS",methods=["GET","POST"])
def DBS():
    return(render_template("DBS.html"))

@app.route("/DBS_prediction",methods=["GET","POST"])
def DBS_prediction():
    q = float(request.form.get("q"))
    return(render_template("DBS_prediction.html",r=90.2 + (-50.6*q)))

@app.route("/creditability",methods=["GET","POST"])
def creditability():
    return(render_template("creditability.html"))

@app.route("/creditability_prediction",methods=["GET","POST"])
def creditability_prediction():
    q = float(request.form.get("q"))
    r=1.22937616 + (-0.00011189*q)
    r = np.where(r >= 0.5, "yes","no")
    r = str(r)
    return(render_template("creditability_prediction.html",r=r))

@app.route("/text_sentiment",methods=["GET","POST"])
def text_sentiment():
    return(render_template("text_sentiment.html"))

@app.route("/text_sentiment_result",methods=["GET","POST"])
def text_sentiment_result():
    q = request.form.get("q")
    r = textblob.TextBlob(q).sentiment
    return(render_template("text_sentiment_result.html",r=r))

@app.route("/makersuite",methods=["GET","POST"])
def makersuite():
    return(render_template("makersuite.html"))

@app.route("/makersuite_1",methods=["GET","POST"])
def makersuite_1():
    q = "Can you help me prepare my tax return?"
    r = model.generate(q)
    return(render_template("makersuite_1_reply.html",r=r.text))

@app.route("/makersuite_gen",methods=["GET","POST"])
def makersuite_gen():
    q = request.form.get("q")
    r = model.generate(q)
    return(render_template("makersuite_gen_reply.html",r=r.text))



@app.route("/text_classification",methods=["GET","POST"])
def text_classification():
    return(render_template("text_classification.html"))

def get_bert_embeddings(text_list):
    tokens = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = bert_embedding_model(**tokens)
    return outputs.last_hidden_state[:, 0, :].numpy()  # CLS token representation

@app.route("/text_classification_result",methods=["GET","POST"])
def text_classification_result():
    q = request.form.get("q")
    input_embedding = get_bert_embeddings([q])
    prediction = bert_model.predict(input_embedding)
    r = 'spam' if prediction[0] == 1 else 'ham'
    return(render_template("text_classification_result.html",r=r))

if __name__ == "__main__":
    app.run()

