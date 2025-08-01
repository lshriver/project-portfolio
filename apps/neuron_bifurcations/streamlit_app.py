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
    
    if selected_key != st.session_state.selected_model:
      st.session_state.selected_model = selected_key
      st.rerun()

    # Initialize neruon model
    neuron_model = NeuronModel(selected_key)

    # Parameter controls
    st.sidebar.markdown("<h2 class='gradient_text1'>Model Parameters</h2>", unsafe_allow_html=True)
    params = {}
    for param_name, (default_val, param_range, description) in neuron_model.get_parameters().items():
      params[param_name] = st.slider(
        f"{param_name} - {description}",
        min_value=param_range[0],
        max_value=param_range[1],
        value=default_val,
        step=(param_range[1] - param_range[0]) / 100
      )

    # Time range
    st.sidebar.markdown("<h2 class='gradient_text1'>Time Configuration</h2>", unsafe_allow_html=True)
    t_max = st.slider("Maximum time", 0.5, 1.0, 20.0, 100.0)
    num_points = st.slider("Number of points", 10, 100, 1000, 5000)

    # Initial Conditions
    st.sidebar.markdown("<h2 class='gradient_text1'>Initial Conditions</h2>", unsafe_allow_html=True)
    num_trajectories = st.slider("Number of trajectories", 1, 3, 10)

    initial_conditions = []
    var_names = neuron_model.get_variable_names()

    # Set reasonable default ranges based on neruon model
    if selected_key == 'fitzhugh_nagumo':
      default_range = [(-3, 3), (-3, 3)]
    elif selected_key == 'hodgkin_huxley':
      default_range = [(-80, 20), (0, 1)]
    elif selected_key == 'morris_lecar':
      default_range = [(-80, 40), (0, 1)]
    elif selected_key == 'izhikevich':
      default_range = [(-80, 30), (-20, 20)]
    elif selected_key == 'wilson_cowan':
      default_range = [(0, 1), (0, 1)]
    elif selected_key == 'integrate_fire':
      default_range = [(-80, -40), (-5, 5)]
    else:
      default_range = [(-2, 2), (-2, 2)]

    if len(var_names) == 2:
      for i in range(num_trajectories):
        st.write(f"Trajectory {i+1}:")
        col1, col2 = st.columns(2)
        with col1:
          x0 = st.number_input(
            f"{var_names[0]}_0",
            value=np.random.uniform(default_range[0][0], default_range[0][1]),
            key = f"x0_{i}"
          )
        with col2:
          y0 = st.number_input(
            f"{var_names[1]}_0",
            value=np.random.uniform(default_range[0][0], default_range[0][1]),
            key = f"y0_{i}"
          )
        initial_conditions.append([x0, y0])
    
    

  parent_utils_style.load_footer()

if __name__ == "__main__":
  main()