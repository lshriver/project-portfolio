import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import odeint
import io
import base64
from ode_systems import Neuron model
from visualization import PhasePortraitPlotter, TrajectoryPlotter
from bifurcation_analysis import BifurcationAnalyzer
import os

# Absolute path to current file 
base_path = os.path.dirname(__file__)

# Path to global static folder
icon_file_path = os.path.join(base_path,"..","..","static","images","favicon.png")

# Normalize the path
icon_file_path = os.path.abspath(icon_file_path)

# Configure page
st.set_page_config(
  page_title = "Neuron Dynamics Visualizer",
  page_icon = "icon_file_path",
  layout = "wide",
  initial_sidebar_state="expanded"
)

def main():
  st.title("Neuron Dynamics Visualizer")

if __name__ == "__main__":
  main()
