import pathlib

import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from SessionState import _get_state


import tensorflow_hub as hub


def main():

    # Needed to clean text_area st object
    state = _get_state()

    # Add web page title
    st.set_page_config(page_title="ITC ML ChatBot")

    # Load model (as st.cache, so it will load only first time execution)
    # and get variables that remain constant for every new query
    embed, unique_questions, question_orig_encodings, lucky_questions, data_pd = load_model()

    # Load first section page and get text filled by user
    input_text = load_page(state, lucky_questions)

    # Convert string text into list to make compatible with existing algorithm
    test_questions = [input_text]

    # Once new text is input new calculations are made
    if test_questions[0] != '':
        # Create encodings for test questions
        question_encodings = embed(test_questions)

        # Make inner product between new question and previous ones (all word embeddings are unitary)
        use_result = np.inner(question_encodings, question_orig_encodings)

        # Calculate maximum cosine similarity
        best_value = np.argmax(use_result[0])

        # Load first section answer (Introduction and tag outputs)
        make_response(best_value)

        # From similar questions obtained, get their activities as proposed policies
        response_list = data_pd[data_pd['Issue'].str.contains(unique_questions[best_value],
                                                              regex=False)]['Activity'].tolist()

        # Show proposed policies
        for i, response_text in enumerate(response_list):
            st.markdown('*Policy {}*:'.format(i+1))
            st.markdown(response_text)

        # Define viz button logic
        if st.button('Show similarity viz'):
            make_viz(best_value, use_result)

            if st.button('Close similarity viz'):
                st.pyplot()

    # Load last section page
    load_footer()

    # Need as last order for well working reset button
    state.sync()


@st.cache
def load_model():
    """
    Load model (as st.cache, so it will load only first time execution)
    and get variables that remain constant for every new query

    Those variables need to be previously computed and saved in a pickle file
    in order to minimize calculations in our web
    """
    # Load module containing USE
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

    # Load previous data from pickle
    pickle_name = 'USE_inputs_2021-11-04_204358.bak'
    with open('./data/' + pickle_name, 'rb') as file_open:
        unique_questions, question_orig_encodings, lucky_questions, data_pd, _ = pickle.load(file_open)

    return embed, unique_questions, question_orig_encodings, lucky_questions, data_pd


def load_page(state, lucky_questions):
    """
    Load first section page: sidebar text, text_area (with titles) and reset+random buttons.

    Return text inside text_are as string
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

    # Show ITC logo from wikipedia
    st.markdown('<img style="float: left; width:20%; heigh:auto" src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/International_Trade_Centre_Logo.svg/1200px-International_Trade_Centre_Logo.svg.png" />',
                unsafe_allow_html=True)

    # Load sidebar message
    st.sidebar.markdown(text_long)

    # Show text_area with title and user information
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

    # Define two page columns in order to show both buttons at same line
    left_column, right_column = st.beta_columns(2)

    # Define Reset button logic
    with left_column:
        if st.button('Clear Text'):
            state.input = ''

    # Define random text button logic
    with right_column:
        if st.button("I'm Feeling Lucky"):
            state.input = lucky_questions[np.random.randint(len(lucky_questions))]

    return state.input


def make_response(best_value):
    """
    -----------
    TODO
    Need to implement tags outputs
    -----------
    Load first section answer (Introduction and tag outputs) and show it
    """

    response_text = """
    It seems you have a **<best_issue_tag>** issue. From our experience, we recommend to you to 
    take some **<best_activity_tag>** policy.
    
    Above we show you some detailed policies that can fit to your issue.    
    """

    # Show first section answer
    st.markdown(response_text)


def make_viz(best_value, use_result):
    """
    Make cosine similarity visualization marking similar question obtained
    """
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
    """
    Load last section page
    """

    text_short = """
    __Disclamer__:


    _This website is not for legal purposes and it is work in progress!_


    This website uses a machine learning model to generate trade policies based on
    previous ITC policies official documents made for similar trade issues detected
    """

    # Show footer section with markdown separator
    st.markdown('---')
    st.markdown(text_short)


if __name__ == "__main__":
    main()