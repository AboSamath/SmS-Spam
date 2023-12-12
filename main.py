import streamlit as st
from PIL import Image
import base64


from utils import background


background('./back.jpg')

st.markdown('<h1 style="background-color: white; color: #f4672f; border: 8px solid #f4672f; padding: 15px; text-align: center">SMS Spam Detection</h1>', unsafe_allow_html=True)

st.markdown("")

st.markdown("")

st.markdown("This application is designed detects/classifies a SMS into SPAM or HAM (normal) based on the textual data using Natural Language Processing.")

st.markdown('#### Please input your SMS !')

texte_utilisateur = st.text_area(" ")

resultat = ""

