"""
# My first app
Here's our first attempt at using data to create a table:
"""

import pandas as pd
import streamlit as st
import numpy as np

df = pd.DataFrame({'first column': [1, 2, 3], 'second column': [4, 5, 6]})

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option