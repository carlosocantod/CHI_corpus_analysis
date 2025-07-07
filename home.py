import streamlit as st
from settings import APP_NAME

st.set_page_config(page_title=APP_NAME, page_icon="🏗️", layout="wide")
st.title(APP_NAME)
st.markdown("Welcome to the CHI Papers Explorer. Use the sidebar to navigate.")
