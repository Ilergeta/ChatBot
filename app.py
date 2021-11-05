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
from SessionState import _get_state
import time

import tensorflow as tf
import tensorflow_hub as hub

def main():

    state = _get_state()

    st.set_page_config(page_title="ITC ML ChatBot")

    embed, unique_questions, question_orig_encodings, lucky_questions, data_pd = load_model()

    input_text = load_page(state, lucky_questions)

    test_questions = [input_text]

    if test_questions[0] != '':
        # Create encodings for test questions
        question_encodings = embed(test_questions)

        use_result = np.inner(question_encodings, question_orig_encodings)

        best_value = np.argmax(use_result[0])

        make_response(best_value)

        response_list = data_pd[data_pd['Issue'].str.contains(unique_questions[best_value], regex=False)]['Activity'].tolist()

        st.markdown('---')

        for response_text in response_list:
            st.markdown(response_text)
            st.markdown('---')

        if st.button('Show similarity viz'):
            make_viz(best_value, use_result)

            if st.button('Close similarity viz'):
                st.pyplot()

    load_footer()

    state.sync()


def load_page(state, lucky_questions):
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

    state.input = st.text_area("Please explain to us your issue, trying to be as concise as possible",
      state.input or '',
      key='input_text',
      height=200,
      max_chars=5000
    )

    left_column, right_column = st.beta_columns(2)

    with left_column:
        if st.button('Clear Text'):
            state.input = ''

    with right_column:
        if st.button("I'm Feeling Lucky"):
            state.input = lucky_questions[np.random.randint(len(lucky_questions))]

    return state.input

@st.cache
def load_model():
    # Load module containing USE
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    # Load previous data from pickle
    pickle_name = 'USE_inputs_2021-11-04_204358.bak'
    with open('./data/' + pickle_name, 'rb') as file_open:
        unique_questions, question_orig_encodings, lucky_questions, data_pd, _ = pickle.load(file_open)
    return embed, unique_questions, question_orig_encodings, lucky_questions, data_pd

def make_response(best_value):
    response_text = """
    It seems you have a **<best_issue_tag>** issue. From our experience, we recommend to you to 
    take some **<best_activity_tag>** policy.
    
    Above we show you some detailed policies that can fit to your issue.    
    """

    st.markdown(response_text)

def make_viz(best_value, use_result):
    # Make visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Similarity values for USE model', fontsize=18)
    ax.set_xlabel('# previous trade policy')
    ax.set_ylabel('cosine similarity')
    x = range(use_result.shape[1])
    ax.scatter(x=x, y=use_result[0, :])
    ax.scatter(x[best_value], use_result[0, best_value], marker='o', s=100)
    ax.text(x[best_value] + 10, use_result[0, best_value], best_value, size=16,
            color='darkorange', weight='bold')
    st.pyplot(fig)

def load_footer():
    text_short = """
    __Disclamer__:


    _This website is not for legal purposes and it is work in progress!_


    This website uses a machine learning model to generate trade policies based on
    previous ITC policies official documents made for similar trade issues detected
    """

    st.markdown('---')
    st.markdown(text_short)


if __name__ == "__main__":
    main()