"""
# My first app
Here's our first attempt at using data to create a table:
"""

import pandas as pd
import streamlit as st
import numpy as np

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared are', x * x)