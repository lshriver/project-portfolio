import streamlit as st
import base64
import os
from pathlib import Path

css_files = [
    Path("static/css/style.css"),
    Path("static/css/theme.css"),
    Path("static/css/streamlit_style.css"),
    Path("static/css/buttons/buttons_rainbow.css")
]

def load_custom_css():
    for css_file in css_files:
        if css_file.exists():
            st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)
        else:
            st.warning(f"CSS file not found: {css_file}")

def get_img_as_base64(fp: str) -> str:
    with open(fp, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
def apply_background(image_path: str):
    if os.path.exists(image_path):
        img_b64 = get_img_as_base64(image_path)
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{img_b64}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-position: center;
            }}
            </style>
        """, unsafe_allow_html=True)

def load_footer(path: str = "templates/footer.html"):
    """Read footer.html and render it at the bottom of the Streamlit app."""
    footer_file = Path(path)
    if not footer_file.exists():
        st.error(f"Footer file not found: {path}")
        return
    html = footer_file.read_text(encoding="utf-8")
    # unsafe_allow_html=True is required to keep our <style> and <div>
    st.markdown(html, unsafe_allow_html=True)