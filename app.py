"""
# My first app
Here's our first attempt at using data to create a table:
"""

import pandas as pd
import streamlit as st
import numpy as np
import time

text_short = """
__Disclamer__:


_This website is not for legal purposes and it is work in progress!_


This website uses a machine learning model to generate trade policies based on
previous ITC policies official documents made for similar trade issues detected
"""

text_long = """
__ITC ML similarity model__:


_This website is not for legal purposes and it is work in progress!_


This website uses a machine learning model to generate trade policies based on
previous ITC policies official documents made for similar trade issues detected.

__Description__:


This project uses a USE (Universal Sentence Encoder) published by Google at [TensorFlow
Hub](https://tfhub.dev/google/universal-sentence-encoder/4) and it is based on a DAN (Deep Averaging Network) architecture. Once new input text
is encoded it is compared (with [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) ) to previous ITC policies and
model outputs more similar activities proposed for those issues.
"""

st.sidebar.markdown(text_long)

st.title('ITC ML ChatBot')

st.text_area(
  "Tell us your trade problem:",
  key='input_text',
  height=200,
  max_chars=5000
)

st.markdown('<h2 style="font-family:Courier;text-align:center;">Your Story</h2>',
        unsafe_allow_html=True,)

st.markdown('---')
st.markdown(text_short)