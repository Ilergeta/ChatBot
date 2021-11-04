"""
# My first app
Here's our first attempt at using data to create a table:
"""

import pandas as pd
import streamlit as st
import numpy as np

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option