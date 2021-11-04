"""
# My first app
Here's our first attempt at using data to create a table:
"""
import pathlib

import os
import pandas as pd
import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

import tensorflow as tf
import tensorflow_hub as hub

def main():

    input_text = load_page()

    st.markdown('<h5 style="font-family:Courier;text-align:center;">'+input_text+'</h5>',
            unsafe_allow_html=True,)

    test_questions = [input_text]

    # Load module containing USE
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

    # Create encodings for test questions
    question_encodings = embed(test_questions)

    # Load previous data from pickle
    pickle_name = 'USE_inputs_2021-11-04_204358.bak'
    with open('./data/'+pickle_name, 'rb') as file_open:
        _, question_orig_encodings, _, data_pd, _ = pickle.load(file_open)

    use_result = np.inner(question_encodings, question_orig_encodings)

    best_value = np.argmax(use_result[0])

    use_result

    # Make visualization
    fig, ax = plt.subplots(figsize=(20, 8))

    fig.title('Similarity for USE model', fontsize=30)

    use_result = np.inner(question_encodings, question_orig_encodings)

    x = range(use_result.shape[1])

    ax.scatter(x=x, y=use_result[0, :])

    ax.scatter(x[best_value], use_result[0, best_value], marker='o', s=100)
    ax.text(x[best_value] + 10, use_result[0, best_value], best_value, size=16,
          color='darkorange', weight='bold')

    st.pyplot(fig)





def load_page():
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

    st.markdown('<h2 style="font-family:Arial;text-align:center;">ITC ML TradeBot</h2>',
            unsafe_allow_html=True,)

    st.markdown('<h3 style="font-family:Arial;text-align:left;">Tell us your trade problem</h4>',
            unsafe_allow_html=True,)

    st.text_area("Please explain to us your issue, trying to be as concise as possible",
      help='Write in the text box and press Ctrl+Enter when finished',
      key='input_text',
      height=200,
      max_chars=5000
    )

    st.markdown('---')
    st.markdown(text_short)

    return st.session_state.input_text

if __name__ == "__main__":
    main()