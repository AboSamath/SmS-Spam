import streamlit as st
from PIL import Image
import base64
from keras.models import load_model
import numpy as np
import joblib

from utils import background, predict_spam


background('./Images/spam3.jpg')

st.markdown('<h1 style="background-color: white; color: #000000; border: 15px solid #990000; padding: 25px; text-align: center">SMS Spam Detection</h1>', unsafe_allow_html=True)

st.markdown("")

st.markdown("")

st.markdown('<h6 style="background-color: white; color: #000000; border: 3px solid #990000; padding: 10px; text-align: center">This application is designed to detects/classifies a SMS into SPAM or HAM (normal) based on the textual data using Natural Language Processing. </h6>', unsafe_allow_html=True)


st.markdown("")

st.markdown('<h6 style="background-color: white; color: #000080; padding: 10px; text-align: center">Please Input a SMS example. </h6>', unsafe_allow_html=True)

texte_utilisateur = st.text_area(" ")


if st.button("SPAM CHECKER"):
    prediction = predict_spam(texte_utilisateur)
    if prediction == 1:
        st.error("Spam detected! Be careful !")
    else:
        st.success("Non Spam (Ham) Safe Message !")
