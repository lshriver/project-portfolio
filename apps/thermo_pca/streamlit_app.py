import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import importlib.util
import os

parent_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "..",      # up to apps/
                 "..",      # up to project-portfolio/
                 "utils",   # the parent-level utils
                 "style.py")
)
spec = importlib.util.spec_from_file_location("parent_utils_style", parent_utils_path)
parent_utils_style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_utils_style)

parent_utils_style.load_custom_css()
parent_utils_style.apply_background("static/images/wisp.jpg")

colormaps_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "..", "..", "utils", "colormaps.py")
)
colormaps_spec = importlib.util.spec_from_file_location("parent_utils_colormps", colormaps_path)
parent_utils_colormaps = importlib.util.module_from_spec(colormaps_spec)
colormaps_spec.loader.exec_module(parent_utils_colormaps)

# Configure page
st.set_page_config(
    page_title = "Thermodynamics and PCA",
    page_icon = "static/images/ember.png",
    layout = "wide",
    initial_sidebar_state="expanded"
)

def main():
    st.markdown("<h1 class='gradient_text1'> Thermodynamics and PCA Analysis </h1>", unsafe_allow_html=True)
    st.markdown("<div class='feature-box'> \
                    <div class='feature-box-content'> \
                        <p class='gradient_text1'> \
                            Statistical mechanics simulation with Boltzmann statistics and PCA. \
                        </p> \
                    </div> \
                </div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()