import pathlib

import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from SessionState import _get_state


import tensorflow_hub as hub


def main():

    # Define main parameters
    tol = 0.1
    max_sentences = 10
    #debug = True

    # Needed to clean text_area st object
    state = _get_state()

    # Add web page title
    st.set_page_config(page_title="ITC ML ChatBot")

    # Load model (as st.cache, so it will load only first time execution)
    # and get variables that remain constant for every new query
    model_tuple = load_model()
    embed, unique_questions, question_orig_encodings, lucky_questions, data_pd, unique_issues_pd = model_tuple

    # Load first section page and get text filled by user
    input_text, debug = load_page(state, lucky_questions)

    # Convert string text into list to make compatible with existing algorithm
    test_questions = [input_text]

    # Once new text is input new calculations are made
    if test_questions[0] != '':
        # Load model (as st.cache, so it will load only first time execution)
        # and get variables that remain constant for every new query
        embed, unique_questions, question_orig_encodings, _, data_pd, unique_issues_pd = model_tuple

        # Create encodings for test questions
        question_encodings = embed(test_questions)

        # Make inner product between new question and previous ones (all word embeddings are unitary)
        use_result = np.inner(question_encodings, question_orig_encodings)

        # Find most similar issue and nearest issues within tolerance
        near_answers = find_near_answers(use_result[0,], tol)

        # Number of sentences in response output
        n_sentences_response = 0

        # Inizilize response list to save activity texts
        response_list = []

        # Inizilize activity id list to avoid outputting duplicated activities
        activity_id_list = []

        for n_answer, answer in enumerate(near_answers):

            issue_id = unique_issues_pd.iloc[answer[0]]['issue_id']

            activity_answer_pd = data_pd[['activity_id', 'activity']][data_pd['issue_id'] == issue_id]

            if debug:
                if n_answer == 0:
                    st.markdown('#' * 100)
                    st.markdown('Considered parameters:')
                    st.markdown('* Tolerance value: {:.2f}%'.format(tol * 100))
                    st.markdown('* Maximum sentences in response output: {}'.format(max_sentences))
                    st.markdown('#' * 100)

                    intro_text = 'Most similar '
                else:
                    intro_text = 'Nearest '

                st.markdown('**' + intro_text + 'issue id:** {} -> similarity: {:.2f}% ({:.2f}%)'
                      .format(issue_id, answer[1] * 100, answer[2] * 100))
                st.markdown('Activities associated: {}'.format(activity_answer_pd.shape[0]))

            for _, row in activity_answer_pd.iterrows():

                activity_id = row['activity_id']
                activity_text = row['activity']

                activity_length = len(activity_text.split('\n'))

                if (n_answer == 0 or ((n_sentences_response + activity_length) <= max_sentences)) \
                        and activity_id not in activity_id_list:

                    if debug:
                        st.markdown('\t* activity_id: {} -> activity_sentences:{}'.format(activity_id, activity_length))

                    response_list.append(activity_text)
                    activity_id_list.append(activity_id)
                    n_sentences_response += activity_length

            if n_sentences_response >= max_sentences:

                if debug:
                    st.markdown('**\nSentences limit reached, not more nearest policies will be considered!!!**')
                    st.markdown('#' * 100)

                break

        # Load first section answer (Introduction)
        intro_text = make_intro()
        st.markdown(intro_text)

        # Show proposed policies
        for i, policy_text in enumerate(response_list):
            st.markdown('*Policy {}*:'.format(i+1))
            st.markdown(policy_text)

        # Define viz button logic
        if st.button('Show similarity viz'):
            make_viz(near_answers[0][0], use_result)

            if st.button('Close similarity viz'):
                st.pyplot()

    # Load last section page
    load_footer()

    # Need as last order for well working reset button
    state.sync()

def find_near_answers(cosine_array, tol=0.1):
    """
    From a 1-dimensional numpy array, find other points with a
    value not more different than "tol"%
    tol value by default: 10%
    """
    # Find maximum value -> issue most similar
    max_value = max(cosine_array)

    # Get indexes from minimum to maximum (except maximum one)
    ordered_indexes = np.argsort(cosine_array).tolist()

    # Define output list
    near_list = [[ordered_indexes.pop(), max_value, 0]]

    # Garantee, at least, one calculation
    difference = 0

    while difference <= tol:
        next_index = ordered_indexes.pop()
        next_value = cosine_array[next_index]

        difference = (max_value-next_value )/max_value

        # In case value is similar enough it appends to list [index, value, difference]
        if difference <= tol:
            near_list.append([next_index, next_value, difference])

    return near_list

def make_intro():
    """
    -----------
    TODO
    Need to implement tags outputs depending on best_value
    -----------
    Load first section answer (Introduction and tag outputs) and show it
    """

    response_text = "Introduction paragraph.\n" +\
    "TO BE DEFINED.\n\n"

    # Show first section answer
    return response_text


@st.cache
def load_model():
    """
    Load model differentiating notebook (script_exec=False) or script exection (script_exec=True)

    Those variables need to be previously computed and saved in a pickle file
    in order to minimize calculations in our web
    """

    # Load module containing USE
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

    # Load previous data from pickle
    pickle_name = 'USE_inputs_2022-01-17_202504.bak'
    with open( 'data/' + pickle_name, 'rb') as file_open:
        unique_questions, question_orig_encodings, lucky_questions, \
        data_pd, _, unique_issues_pd = pickle.load(file_open)

    return embed, unique_questions, question_orig_encodings, lucky_questions, data_pd, unique_issues_pd


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

    # Show debug checkbox
    debug = st.sidebar.checkbox('Debug mode')

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

    return state.input, debug


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