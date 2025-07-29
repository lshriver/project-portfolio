import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import odeint
import io
import base64
from utils.ode_systems import NeuronModel
from utils.visualization import PhasePortraitPlotter, TrajectoryPlotter
from utils.bifurcation_analysis import BifurcationAnalyzer
import importlib.util
import sys, os

parent_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "..",          # up to apps/
                 "..",          # up to project-portfolio/
                 "utils",       # the parent‐level utils
                 "style.py")
)
spec = importlib.util.spec_from_file_location("parent_utils_style", parent_utils_path)
parent_utils_style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_utils_style)

parent_utils_style.load_custom_css()
parent_utils_style.apply_background("static/images/wisp.jpg")


# Configure page
st.set_page_config(
  page_title = "Neuron Dynamics Visualizer",
  page_icon = "static/images/favicon.png",
  layout = "wide",
  initial_sidebar_state="expanded"
)

# Initialize session state
if 'selected_model' not in st.session_state:
  st.session_state.selected_model = 'fitzhugh_nagumo'

def main():
  st.markdown("<h1> 🧠 <span class='gradient_text1'> Neuron Dynamics Visualizer </span> </h1>", unsafe_allow_html=True)
  st.markdown("<span class='gradient_text1'>Interactive bifurcation analysis of neuronal signaling models with phase portraits and dynamics visualization</span>", unsafe_allow_html=True)


  # Sidebar for model selection and parameters
  with st.sidebar: 
    st.sidebar.markdown("<h2 class='gradient_text1'>Neuron Model Configuration</h2>", unsafe_allow_html=True)

    # Model selection
    model_options = {
      'fitzhugh_nagumo': 'FitzHugh-Nagumo',
      'hodgkin_huxley': 'Hodgkin-Huxley (simplified)',
      'morris-lecar': 'Morris-Lecar',
      'izhikevich': 'Izhikevich',
      'wilson_cowan': 'Wilson-Cowan',
      'integrate-fire': 'Integrate-and-Fire (Adaptive)'
    }

    selected_key = st.selectbox("Select Neuron Model", options=list(model_options.keys()),
                                    format_func = lambda x: model_options[x],
                                    index = list(model_options.keys()).index(st.session_state.selected_model)
                                )
    

  parent_utils_style.load_footer()

if __name__ == "__main__":
  main()
