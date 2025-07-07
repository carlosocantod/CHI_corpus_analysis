import streamlit as st
from settings import APP_NAME
from setup_streamlit import load_data


st.set_page_config(page_title=APP_NAME, page_icon="🏗️", layout="wide")
st.title(APP_NAME)
st.markdown("Welcome to the CHI Papers Explorer. Use the sidebar to navigate.")
model, embeddings, metadata, top_words_topic = load_data()
