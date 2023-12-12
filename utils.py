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

