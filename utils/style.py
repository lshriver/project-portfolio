import streamlit as st
import base64
import os
from pathlib import Path

def load_custom_css():
    # Get the project root directory (two levels up from current file)
    project_root = Path(__file__).parent.parent
    
    css_files = [
        project_root / "static/css/style.css",
        project_root / "static/css/theme.css", 
        project_root / "static/css/streamlit_style.css"
    ]
    
    for css_file in css_files:
        if css_file.exists():
            st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)
        else:
            st.warning(f"CSS file not found: {css_file}")

def get_img_as_base64(fp: str) -> str:
    with open(fp, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
def apply_background(image_path: str):
    # Resolve path relative to project root
    project_root = Path(__file__).parent.parent
    full_image_path = project_root / image_path
    
    if full_image_path.exists():
        img_b64 = get_img_as_base64(str(full_image_path))
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
    else:
        st.warning(f"Background image not found: {full_image_path}") 

def load_footer(path: str = "templates/footer.html"):
    """Read footer.html and render it at the bottom of the Streamlit app."""
    # Resolve path relative to project root
    project_root = Path(__file__).parent.parent
    footer_file = project_root / path
    if not footer_file.exists():
        st.error(f"Footer file not found: {footer_file}")
        return
    html = footer_file.read_text(encoding="utf-8")
    # unsafe_allow_html=True is required to keep our <style> and <div>
    st.markdown(html, unsafe_allow_html=True)