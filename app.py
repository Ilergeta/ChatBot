"""
# My first app
Here's our first attempt at using data to create a table:
"""

import pandas as pd
import streamlit as st
import numpy as np

st.text_input("Your name", key="name")

# You can access the value at any point with:
st.session_state.name