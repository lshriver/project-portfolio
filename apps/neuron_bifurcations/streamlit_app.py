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

colormaps_path = os.path.abspath(
  os.path.join(os.path.dirname(__file__),
               "..", "..", "utils", "colormaps.py")
)
colormaps_spec = importlib.util.spec_from_file_location("parent_utils_colormaps", colormaps_path)
parent_utils_colormaps = importlib.util.module_from_spec(colormaps_spec)
colormaps_spec.loader.exec_module(parent_utils_colormaps)

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
  st.markdown("<div class='feature-box'> \
                  <div class='feature-box-content'> \
                    <p class='gradient_text1'> \
                      Interactive bifurcation analysis of neuronal signaling models with phase portraits and dynamics visualization. \
                    </p>  \
                  </div>  \
                </div>", unsafe_allow_html=True)

  # Sidebar for model selection and parameters
  with st.sidebar: 
    st.sidebar.markdown("<h2 class='gradient_text1'>Neuron Model Configuration</h2>", unsafe_allow_html=True)

    seed = st.sidebar.number_input("Random seed", 0 ,10000, 27)
    np.random.seed(seed)

    # Model selection
    model_options = {
      'fitzhugh_nagumo': 'FitzHugh-Nagumo',
      'hodgkin_huxley': 'Hodgkin-Huxley (simplified)',
      'morris_lecar': 'Morris-Lecar',
#      'izhikevich': 'Izhikevich',
#      'wilson_cowan': 'Wilson-Cowan',
#      'integrate_fire': 'Integrate-and-Fire (Adaptive)'
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
    st.sidebar.markdown("<h2 class='gradient_text1'>", unsafe_allow_html=True)
    with st.expander("Parameters"):
      params = {}
      for param_name, (default_val, param_range, description) in neuron_model.get_parameters().items():
        params[param_name] = st.slider(
          rf"{description}",
          min_value=param_range[0],
          max_value=param_range[1],
          value=default_val,
          step=(param_range[1] - param_range[0]) / 100,
          key=f"param_{param_name}"
        )
    st.sidebar.markdown("</h2>", unsafe_allow_html=True)
    

    # Time range
    # st.sidebar.markdown("<h2 class='gradient_text1'>Time Configuration</h2>", unsafe_allow_html=True)
    with st.expander("Time Settings"):
      t_max = st.slider("Maximum time", 20.0, 500.0, 100.0, 0.5, key="time_max")
      num_points = st.slider("Number of points", 100, 5000, 1000, 100, key="num_points")
    st.sidebar.markdown("</h2>", unsafe_allow_html=True)

    # Initial Conditions
    st.sidebar.markdown("<h2 class='gradient_text1'>Initial Conditions</h2>", unsafe_allow_html=True)
    num_trajectories = st.slider("Number of trajectories", min_value=1, value=3, max_value=10, key="num_trajectories")

    initial_conditions = []
    var_names = neuron_model.get_variable_names()

    # Set reasonable default ranges based on neruon model
    if selected_key == 'fitzhugh_nagumo':
      default_range = [(-3.0, 3.0), (-3.0, 3.0)]
    elif selected_key == 'hodgkin_huxley':
      default_range = [(-80.0, 20.0), (0.0, 1.0)]
    elif selected_key == 'morris_lecar':
      default_range = [(-80.0, 40.0), (0.0, 1.0)]
#    elif selected_key == 'izhikevich':
#      default_range = [(-80, 30), (-20, 20)]
#    elif selected_key == 'wilson_cowan':
#      default_range = [(0.0, 1.0), (0.0, 1.0)]
#    elif selected_key == 'integrate_fire':
#      default_range = [(-80.0, -40.0), (-5.0, 5.0)]
    else:
      default_range = [(-2.0, 2.0), (-2.0, 2.0)]

    if len(var_names) == 2:
      for i in range(num_trajectories):
        st.markdown(f"**Trajectory {i+1}**")    
        
        col1, col2 = st.sidebar.columns(2)

        x0 = col1.number_input(
          label = fr"${var_names[0]}_0$",
          min_value = default_range[0][0],
          max_value = default_range[0][1],
          value = float(np.random.uniform(*default_range[0])),
          key = f"x0_{i}"
        )

        y0 = col2.number_input(
          label = rf"${var_names[1]}_0$",
          min_value = default_range[1][0],
          max_value = default_range[1][1],
          value = float(np.random.uniform(*default_range[1])),
          key = f"y0_{i}"
        )

        initial_conditions.append([x0, y0])

  # Main content area
  tab1, tab2, tab3, tab4 = st.tabs(["Phase Portrait", "Bifurcation Analysis", "Spike Analysis", "Model Info"])

  with tab1:
    st.markdown("<h2 class='gradient_text2'>Phase Portrait & Neural Dynamics</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
      # Generate phase portrait
      if len(var_names) >- 2:
        plotter = PhasePortraitPlotter(neuron_model)
        fig_phase = plotter.plot_phase_portrait(
          params, 
          initial_conditions,
          t_max,
          num_points
        )
        plotter._add_nullclines_2d(
          fig_phase,
          params,
          var_names
        )
        st.plotly_chart(fig_phase, use_container_width=True)

    with col2:
      # Generate time series
      trajectory_plotter = TrajectoryPlotter(neuron_model)
      fig_time = trajectory_plotter.plot_time_series(
        params,
        initial_conditions[0] if initial_conditions else [1, 1],
        t_max, 
        num_points
      )
      st.plotly_chart(fig_time, use_container_width=True)

    # Show neural activity data
    if st.checkbox("Show nerual activity data"):
      t = np.linspace(0, t_max, num_points)
      for i, ic in enumerate(initial_conditions[:3]):  # Limit to 3 for display
        sol = odeint(lambda y, t: neuron_model.equations(y, t, params), ic, t)
        df_data = {
          'Time (ms)': t,
        }
        for j, var_name in enumerate(var_names):
          df_data[f'{var_name}_{i+1}'] = sol[:, j]

        if i == 0:
          import pandas as pd
          df = pd.DataFrame(df_data)
        else:
          for j, var_name in enumerate(var_names):
            df[f'{var_name}_{i+1}'] = sol[:,1]

      st.dataframe(df.head(20))

      # Export functionality
      csv = df.to_csv(index=False)
      st.download_button(
        label="Download neural activity data as CSV",
        data=csv,
        file_name=f"{selected_key}_neural_activty.csv",
        mime="text/csv"
      )

  with tab2:
    st.markdown("<h2 class='gradient_text2'>Neural Bifurcation Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<div class='feature-box'> \
                  <div class='feature-box-content'> \
                    <p class='gradient_text1'> \
                      Analyze how neural dynamics change as parameters vary. This is crucial for understanding excitability thresholds, oscillatory behavior, and transitionts between firing patterns. \
                    </p>  \
                  </div>  \
                </div>", unsafe_allow_html=True)

    # Bifurcation parameter selection
    param_names = list(params.keys())
    if param_names:
      bifurcation_param = st.selectbox("Select Bifurcation Parameter", param_names)

      col1, col2 = st.columns(2)
      with col1:
        param_min = st.number_input(
          f"Minimum {bifurcation_param}",
          value=max(0.01, params[bifurcation_param] - 2)
        )    
      with col2:
        param_max = st.number_input(
          f"Maximum {bifurcation_param}",
          value=params[bifurcation_param] + 2
        )

      num_param_points = st.slider("Parameter resolution", 50, 500, 150, key="param_resolution")

      if st.button("Generate Bifurcation Diagram"):
        with st.spinner("Computing neural bifurcation diagram..."):
          analyzer = BifurcationAnalyzer(neuron_model)

          # Create modified params for bifurcation analysis
          bifurcation_params = params.copy()
          fig_bifurcation = analyzer.plot_bifurcation_diagram(
            bifurcation_params,
            bifurcation_param,
            param_min,
            param_max,
            num_param_points,
            initial_conditions[0] if initial_conditions else [1, 1],
            t_max
          )

          st.plotly_chart(fig_bifurcation, use_container_width=True)

          # Add interpretation for neural model context
          st.markdown("<h4 class='gradient_text1'>Interpretation</h4>", unsafe_allow_html=True)
          if selected_key == 'fitzhugh_nagumo':
            st.write("Look for trasitions between excitable (single stable point) and oscilaltory (limit cycle) regimes.")
          elif selected_key == 'hodgkin_huxley':
            st.write("Observe threshold behavior and transitions to repetitive spiking.")
          elif selected_key == 'morris_lecar':
            st.write("Morris-Lecar shows rich bifurcation structure including saddle-node and Hopf bifurcations.")
#          elif selected_key =='izhikevich':
#            st.write("Different parameter regions produce various neural firing patterns (regular, bursting, chattering, etc.).")
#          elif selected_key == 'wilson_cowan':
#            st.write("Population dynamics can show multistability and osicllations.")
          else:
            st.write("Bifurcation diagram shows paramete regimes with different dynamical behaviors.")

    # Neural excitability analysis
    st.markdown("<h2 class='gradient_text1'>Excitability Analysis</h2>", unsafe_allow_html=True)
    if st.button("Analyze Neural Excitability"):
      with st.spinner("Analyzing neural excitability..."):
        # Test response to brief current pulses
        t_pulse = np.linspace(0, 50, 2000)
        pulse_amplitudes = np.linspace(0, params.get('I', 1) * 3, 20)

        fig_excitability =go.Figure()

        for i, pulse_amp in enumerate(pulse_amplitudes[::3]):   # Sample every 3rd
          # Modify current for pulse
          pulse_params = params.copy()
          if 'I' in pulse_params:
            pulse_params['I'] = pulse_amp
            
          try:
            sol_pulse = odeint(
              lambda y, t: neuron_model.equations(y, t, pulse_params), 
              initial_conditions[0] if initial_conditions else [-70, 0],
              t_pulse
            )

            colors = ["#4C48F2", "#5E17EB", "#EB0FD5", "#FF8855", "#70e000", "#38b000", "#00e8ff", "#2F8DF7"]

            fig_excitability.add_trace(go.Scatter(
              x=t_pulse,
              y=sol_pulse[:, 0],    # Voltage, first variable
              mode='lines',
              name=f'I={pulse_amp:.2f}',
              line=dict(
                width=2,
                color = colors[i % len(colors)]
              ),
            ))

          except:
            continue
        
        fig_excitability.update_layout(
          title="Neural Response to Current Steps",
          xaxis_title="Time",
          yaxis_title=var_names[0],
          height=400,
          paper_bgcolor='rgba(0, 0, 0, 0.5)',
          plot_bgcolor = 'rgba(0, 0, 0, 0.5)',
          margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_excitability, use_container_width=True)

  with tab3:
    st.markdown("<h3 class='gradient_text1'>Spike Train Analysis</h3>", unsafe_allow_html=True)
    st.markdown("<div class='feature-box'> \
                  <div class='feature-box-content'> \
                    <p class='gradient_text1'> \
                      Analysis of neural firing patterns and spike timing. \
                    </p>  \
                  </div>  \
                </div>", unsafe_allow_html=True)
    
    # Generate longer time series for spike analysis
    t_long = np.linspace(0, t_max * 2, num_points * 2)

    if st.button("Analyze Spike Patterns"):
      with st.spinner("Computing spike train analysis..."):
        try:
          sol_long = odeint(
            lambda y, t: neuron_model.equations(y, t, params),
            initial_conditions[0] if initial_conditions else [-70, 0],
            t_long
          )

          voltage = sol_long[:, 0]

          # Detect spikes (simple threshold crossing)
          threshold = np.mean(voltage) + 2 * np.std(voltage)
          spike_times = []

          for i in range(1, len(voltage)):
            if voltage[i-1] < threshold and voltage[i] >= threshold:
              spike_times.append(t_long[i])

          colormap_blue = ["#00e8ff", "#14b5ff", "#3a98ff", "#0070eb"]

          # Plot voltage trace with spike detection
          fig_spikes = go.Figure()
          fig_spikes.add_trace(go.Scatter(
            x=t_long,
            y=voltage,
            mode='lines',
            name='Membrane potential',
            line=dict(
              color = colormap_blue[i % len(colormap_blue)],
              width=2
            )
          ))

          # Mark spike times
          if spike_times:
            fig_spikes.add_trace(go.Scatter(
              x=spike_times,
              y=[threshold] * len(spike_times),
              mode='markers',
              name='Spikes',
              marker=dict(
                color = "#38b000", size=8, symbol='triangle-up'
              )
            ))

          fig_spikes.add_hline(y=threshold, line_dash="dash", line_color="#0070eb",
                              annotation_text=f"Threshold = {threshold:.1f}")
          
          fig_spikes.update_layout(
            title="Spike Detection",
            xaxis_title="Time",
            yaxis_title=var_names[0],
            height=400,
            paper_bgcolor='rgba(0, 0, 0, 0.5)',
            plot_bgcolor = 'rgba(0, 0, 0, 0.5)',
            margin=dict(l=40, r=40, t=40, b=40)
          )

          st.plotly_chart(fig_spikes, use_container_width=True)

          # Spike  statistics
          if len(spike_times) > 1:
            isi = np.diff(spike_times)    # Inter-spike intervals
            firing_rate = len(spike_times) / (t_long[-1] - t_long[0]) * 1000  # Hz

            col1, col2, col3 = st.columns(3)
            with col1:
              st.metric("Number of Spikes", len(spike_times))
            with col2:
              st.metric("Firing Rate", f"{firing_rate:.1f} Hz")
            with col3:
              st.metric("Mean ISI", f"{np.mean(isi):.2f} ms")

            # ISI histogram
            if len(isi) > 2:
              fig_isi = go.Figure(
                data=go.Histogram(x=isi, nbinsx=20)
              )
              fig_isi.update_layout(
                title="Inter-Spike Interval Distribution",
                xaxis_title="ISI (ms)",
                yaxis_title="Count",
                height=300,
                paper_bgcolor='rgba(0, 0, 0, 0.5)',
                plot_bgcolor = 'rgba(0, 0, 0, 0.5)',
                margin=dict(l=40, r=40, t=40, b=40)
              )
              st.plotly_chart(fig_isi, use_container_width=True)
          else:
            st.info("No spikes detected or insufficient spikes for analysis.")
          
        except Exception as e:
          st.error(f"Error in spike analysis: {str(e)}")  

  with tab4:
    st.markdown("<h2 class='gradient_text1'>Neural Model Informaiton</h2>", unsafe_allow_html=True)

    # Display model equations
    st.markdown("<h3 class='gradient_text2'>Current Parameters</h3>", unsafe_allow_html=True)
    equations_latex = neuron_model.get_equations_latex()
    for eq in equations_latex:
      st.latex(eq)

    # Model description
    st.markdown("<h3 class='gradient_text1'>Model Description</h3>", unsafe_allow_html=True)
    description = neuron_model.get_description()
    st.write(description)

    # Neural model specific information
    st.markdown("<h3 class='gradient_text1'>Neurobiological Context</h3>", unsafe_allow_html=True)
    if selected_key == 'fitzhugh_nagumo':
      st.write("**Variables:** $V$ = membrane potential, $W$=recovery variable")
      st.write("**Key Features:** Simplified neuron model showing excitability and oscillations")
    elif selected_key == 'hodgkin_huxley':
      st.write("**Variables:** $V$ = membrane potential (mV), $n$ = potassium ion channel activation")
      st.write("**Key Features:** Classical model for action potential generation")
    elif selected_key == 'morris_lecar':
      st.write("**Variables:** $V$ = membrane potential (mV), $W$ = potassium ion channel activation")
      st.write("**Key Features:** calcium ion and potassium ion channel dynamics, rich bifiurcation structure")
#    elif selected_key == 'izhikevich':
#      st.write("**Variables:** $v$ membrane potential (mV), $u$= recovery variable")
#      st.write("**Key Features:** Efficient model reproducitng various firing patterns")
#    elif selected_key == 'wilson_cowan':
#      st.write("**Variables:** $E$ = excitatory activity, $I$ = inhibitory activity")
#      st.write("**Key Features:** Neural population dynamics, oscillations and waves")
#    elif selected_key == 'integrate_fire':
#      st.write("**Variables:** $V$ = membrane potential (mV), w = adaptation current (nA)")
#      st.write("**Key Features:** Spike-frequency adaptation and realistic firing patterns")

    # Equilibrium points (if available)
    if hasattr(neuron_model, 'get_equilibrium_points'):
      st.markdown("<h2 class='gradient_text1'>Equilibrium Points</h2>", unsafe_allow_html=True)
      eq_points = neuron_model.get_equilibrium_points(params)
      if eq_points:
        for i, point in enumerate(eq_points):
          st.write(f"Equilibrium {i+1}: {point}")
      else:
        st.write("Equilibrium points require numerical computation for this model.")

  parent_utils_style.load_footer()

if __name__ == "__main__":
  main()