import streamlit as st
from PIL import Image
import base64
from keras.models import load_model
import numpy as np
import joblib

from utils import background, predict_spam


background('./Images/spam.jpg')

st.markdown('<h1 style="background-color: white; color: #000080; border: 8px solid #000080; padding: 15px; text-align: center">SMS Spam Detection</h1>', unsafe_allow_html=True)

st.markdown("")

st.markdown("")

st.markdown('<h5 style="background-color: white; color: #000080; border: 8px solid #000080; padding: 10px; text-align: center">This application is designed detects/classifies a SMS into SPAM or HAM (normal) based on the textual data using Natural Language Processing. </h5>', unsafe_allow_html=True)

resultat = ""

#Chargement du modèle
loaded_rf = joblib.load("./Models/Random_Forest.joblib")

st.markdown("")

st.markdown('#### Please input your SMS !')

texte_utilisateur = st.text_area(" ")

if st.button("SPAM Detection"):
    if predict_spam(texte_utilisateur) :
        print("This is a SPAM Message, Be Careful !")
    else:
        print("This is a normal message")


