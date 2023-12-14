import base64
import streamlit as st
from PIL import ImageOps, Image
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from bs4 import BeautifulSoup as bs
from requests import get
import re
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import joblib



def background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()

    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:./jpg;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

sample_message = ""

# Créer un text parser utilisant de tokenisation
parser = PlaintextParser.from_string(sample_message, Tokenizer('english'))

#Chargement du modèle
loaded_rf = joblib.load("./Models/Random_Forest.joblib")

#Cargement du transformateur
load_tfid = joblib.load("./tfid_transformer.joblib")

wnl = WordNetLemmatizer()

corpus = []


def predict_spam(sample_message):
  
  sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_message)
  sample_message = sample_message.lower()
  sample_message_words = sample_message.split()
  sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
  final_message = [wnl.lemmatize(word) for word in sample_message_words]
  final_message = ' '.join(final_message)

  temp = load_tfid.fit_transform([final_message]).toarray()
  feature_names = load_tfid.get_feature_names_out()
  prediction_rf = loaded_rf.predict(temp)
  prediction = prediction_rf.toarray()

  return prediction[0]



