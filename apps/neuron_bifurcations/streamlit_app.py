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
from utils.style import *
import os

load_custom_css()
apply_background("static/images/wisp.jpg")


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

if __name__ == "__main__":
  main()
