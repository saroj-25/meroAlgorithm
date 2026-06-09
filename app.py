import os
import streamlit as st

st.set_page_config(
    page_title="meroAlgorithm",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    css_files = [
        "frontend/styles/main.css",
        "frontend/styles/sidebar.css",
        "frontend/styles/navbar.css",
        "frontend/styles/main_content.css",
        "frontend/styles/right_panel.css",
    ]
    for css_file in css_files:
        if not os.path.exists(css_file):
            st.error(f"CSS file not found: {css_file}")
            continue
        with open(css_file, "r") as f:
            st.html(f"<style>{f.read()}</style>")

load_css()

from frontend.components.sidebar import render_sidebar
from frontend.components.navbar import render_navbar
from frontend.components.main_content import render_main_content
from frontend.components.right_panel import render_right_panel

render_sidebar()

render_navbar()

col_main, col_right = st.columns([2.5, 1], gap="medium")

with col_main:
    render_main_content()

with col_right:
    render_right_panel()