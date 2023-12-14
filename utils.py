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

rf = RandomForestClassifier(n_estimators=10)
tfidf = TfidfVectorizer(max_features=500)
wnl = WordNetLemmatizer()
corpus = []

def Nettoyage(sample_message):
   
   for sms_string in list(sample_message):

    # Nettoyage des caractères spéciaux
    sample_message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms_string)

    # Convertion des caractères en miniscule
    sample_message = sample_message.lower()

    # Tokenisation en mot
    words = sample_message.split()

    # Suppression des mots vides
    filtered_words = [word for word in words if word not in set(stopwords.words('english'))]

    # Lemmatisation des mots
    lemmatized_words = [wnl.lemmatize(word) for word in filtered_words]

    # Jointure des mots lemmatisés
    sample_message = ' '.join(lemmatized_words)

    # Construction d'un corpus de messages
    corpus.append(sample_message)
    

   return sample_message

def predict_spam(sample_message):
  sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_message)
  sample_message = sample_message.lower()
  sample_message_words = sample_message.split()
  sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
  final_message = [wnl.lemmatize(word) for word in sample_message_words]
  final_message = ' '.join(final_message)

  temp = tfidf.transform([final_message]).toarray()
  return rf.predict(temp)


