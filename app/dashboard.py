"""Implementation and Visualization of various Gradient Descent algorithm"""
# import os

import numpy as np
import pandas as pd
import streamlit as st

from pagination.about import About
from pagination.vanilla import Vanilla
from pagination.adagrad import Adagrad
from pagination.adam import Adam
from pagination.momentum import Momentum
from pagination.nag import NAG
from pagination.rmsprop import RMSprop
from utils.sidebar import sidebar_caption

# Config the whole app
st.set_page_config(
    page_title="Gradient Descent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

padding = 0
st.markdown(
    f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """,
    unsafe_allow_html=True,
)
st.markdown(
    """
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


def main():
    """A streamlit app template"""
    # import pdb;pdb.set_trace()
    st.sidebar.image("app/data/sidebar_img.jpg", use_column_width="always")
    st.sidebar.title("Gradient Descent Algorithm")

    PAGES = {
        "Vanilla Gradient Descent": Vanilla,
        "Momentum-based Gradient Descent": Momentum,
        "Nesterov accelerated Gradient Descent": NAG,
        "Adagrad Gradient Descent": Adagrad,
        "RMSprop Gradient Descent": RMSprop,
        "Adam Gradient Descent": Adam,
        
    }

    # Select pages
    # Use dropdown if you prefer
    selection = st.sidebar.radio("Variants of Gradient Descent", list(PAGES.keys()))   
    sidebar_caption()

    page = PAGES[selection]

    DATA = {"base": fake_data()}

    with st.spinner(f"Loading Page {selection} ..."):
        page = page(DATA)
        page()


if __name__ == "__main__":
    main()
