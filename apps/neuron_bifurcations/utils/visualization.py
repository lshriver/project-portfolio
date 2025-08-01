import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import importlib.util
import os

colormaps_path = os.path.abspath(
  os.path.join(os.path.dirname(__file__),
               "..", "..", "..", "utils", "colormaps.py")
)
colormaps_spec = importlib.util.spec_from_file_location("parent_utils_colormaps", colormaps_path)
parent_utils_colormaps = importlib.util.module_from_spec(colormaps_spec)
colormaps_spec.loader.exec_module(parent_utils_colormaps)

class PhasePortraitPlotter:
    """Handles phase portrait visualization for neural models"""
    def __init__(self, neuron_model):
        self.neuron_model = neuron_model

    def plot_phase_portrait(self, params, initial_conditions, t_max, num_points):
        """Create interactive phase portrait with multiple neural trajectories"""
        t = np.linspace(0, t_max, num_points)
        var_names = self.neuron_model.get_variable_names()

        if len(var_names) == 2:
            return self._plot_2d_phase_portrait(params, initial_conditions, t, var_names)
        elif len(var_names) == 3:
            return self._plot_3d_phase_portrait(params, initial_conditions, t, var_names)
        else: 
            raise ValueError("Phase portraits only supported for 2D and 3D systems")
        
    def _plot_2d_phase_portrait(self, params, initial_conditions, t, var_names):
        """Create 2D phase portrait"""
        fig = go.Figure()

        # Plot trajectories
        colors = ["#00E8FF", "#2F8DF7", "#5E91EE", "#8D66E6", "#BC3ADD", "#EB0FD5"]
        colors = ["#4C48F2", "#5E17EB", "#EB0FD5", "#FF8855", "#70e000", "#38b000", "#00e8ff", "#2F8DF7"]

        for i, ic in enumerate(initial_conditions):
            try:
                sol = odeint(
                    lambda y, t: self.neuron_model.equations(y, t, params),
                    ic, t
                )
                
                col = colors[i % len(colors)]

                # Plot trajectory
                fig.add_trace(go.Scatter(
                    x = sol[:, 0],
                    y = sol[:, 1],
                    mode = 'lines',
                    name = f'Trajectory {i+1}',
                    line = dict(color=col, width=2)
                ))

                # Plot initial conditions
                fig.add_trace(go.Scatter(
                    x = [sol[0, 0]],
                    y = [sol[0, 1]],
                    mode = 'markers',
                    name = f'IC {i+1}',
                    marker = dict(color=col, size=8, symbol='circle'),
                    showlegend = False
                ))

                # Plot final point
                fig.add_trace(go.Scatter(
                    x = [sol[-1, 0]],
                    y = [sol[-1, 1]],
                    mode = 'markers',
                    name = f'Final {i+1}',
                    marker = dict(color=col, size=8, symbol='x'),
                    showlegend = False
                ))

            except Exception as e:
                print(f"Error computing trajectory {i+1}: {e}")
                continue

        # Add vector field
        self._add_vector_field_2d(fig, params, var_names)

        # Update Layout
        fig.update_layout(
            title="Phase Portrait",
            xaxis_title = var_names[0],
            yaxis_title = var_names[1],
            hovermode = 'closest',
            showlegend = True,
            width = 600,
            height = 500,
            paper_bgcolor='rgba(0, 0, 0, 0.5)',
            plot_bgcolor = 'rgba(0, 0, 0, 0.5)',
        )

        return fig
    
    def _plot_3d_phase_portrait(self, params, initial_conditions, t, var_names):
        """Create 3D phase portrait"""
        fig = go.Figure()

        colors = ["#00E8FF", "#2F8DF7", "#5E91EE", "#8D66E6", "#BC3ADD", "#EB0FD5"]
        for i, ic in enumerate(initial_conditions):
            try: 
                sol = odeint(lambda y, t: self.neuron_model.equations(y, t, params), ic, t)

                # Plot 3D trajectory
                fig.add_trace(go.Scatter3d(
                    x=sol[:, 0],
                    y=sol[:, 1],
                    z=sol[:, 2],
                    mode='lines',
                    name=f'Trajectory {i+1}',
                    line=dict(color=colors[i % len(colors)], width=4)
                ))

                # Plot initial condition
                fig.add_trace(go.Scatter3d(
                    x=[sol[0, 0]],
                    y=[sol[0, 1]],
                    z=[sol[0, 2]],
                    mode='markers',
                    name=f'IC {i+1}',
                    marker=dict(
                        colors=colors[i % len(colors)],
                        size=6
                    ),
                    showlegend=False
                ))

            except Exception as e:
                print(f"Error computing trajectory {i+1}: {e}")

            # Update layout
            fig.update_layout(
                title="3D Phase Portrait",
                scene=dict(
                    xaxis_title=var_names[0],
                    yaxis_title=var_names[1],
                    zaxis_title=var_names[2]
                ),
                width=600,
                widht=500,
                paper_bgcolor='rgba(0, 0, 0, 0.5)',
                plot_bgcolor = 'rgba(0, 0, 0, 0.5)',
                margin=dict(l=40, r=40, t=40, b=40)
            )

            return fig
        
    def _add_vector_field_2d(self, fig, params, var_names):
        """Add vector field to 2D phase portrait"""

        # Create grid for vector field
        x_range = [-3, 3]
        y_range = [-3, 3]

        # Try to get better bounds from current traces
        if fig.data:
            all_x = []
            all_y = []
            for trace in fig.data:
                if hasattr(trace, 'x') and trace.x is not None:
                    all_x.extend(trace.x)
                if hasattr(trace, 'y') and trace.y is not None:
                    all_y.extend(trace.y)

            if all_x and all_y:
                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = min(all_y), max(all_y)
                x_range = [x_min - 0.5, x_max + 0.5]
                y_range = [y_min - 0.5, y_max + 0.5]

        # Create grid
        x_grid = np.linspace(x_range[0], x_range[1], 15)
        y_grid = np.linspace(y_range[0], y_range[1], 15)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Calculate vector field
        DX = np.zeros_like(X)
        DY = np.zeros_like(Y)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    state = [X[i, j], Y[i, j]]
                    derivatives = self.neuron_model.equations(state, 0, params)
                    DX[i, j] = derivatives[0]
                    DY[i, j] = derivatives[1]
                except:
                    DX[i, j] = 0
                    DY[i, j] = 0

        # Normalize the arrows
        M = np.sqrt(DX**2 + DY**2)
        M[M == 0] = 1   # avoid division by zero
        DX_norm = DX / M
        DY_norm = DY / M

        # Add vector field as sccatter plot with arrows
        scale = 0.1
        for i in range(0, len(x_grid), 2):      # skip some for clarity
            for j in range(0, len(y_grid), 2):
                fig.add_trace(go.Scatter(
                    x=[X[j, i], X[j, i] + scale * DX_norm[j, i]],
                    y=[Y[j, i], Y[j, i] + scale * DY_norm[j, i]],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
class TrajectoryPlotter:
    """Handles neural time series trajectory plotting"""

    def __init__(self, neuron_model):
        self.neuron_model = neuron_model
    
    def plot_time_series(self, params, initial_condition, t_max, num_points):
        """"Create time series plot of variables"""
        t = np.linspace(0, t_max, num_points)
        var_names = self.neuron_model.get_variable_names()

        try: 
            sol = odeint(
                lambda y, t: self.neuron_model.equations(y, t, params),
                initial_condition, t
            )

            fig = go.Figure()
            colors = ["#00E8FF", "#2F8DF7", "#5E91EE", "#8D66E6", "#BC3ADD", "#EB0FD5"]

            # Plot each variable
            for i, var_name in enumerate(var_names):
                fig.add_trace(go.Scatter(
                    x=t,
                    y=sol[:, i],
                    mode='lines',
                    name=var_name,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))

            fig.update_layout(
                title="Time Series",
                xaxis_title="Time",
                yaxis_title="Value",
                hovermode='x unified',
                width=600,
                height=500,
                paper_bgcolor='rgba(0, 0, 0, 0.5)',
                plot_bgcolor = 'rgba(0, 0, 0, 0.5)',
                margin=dict(l=40, r=40, t=40, b=40)
            )
            return fig
        
        except Exception as e:
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error computing trajectory: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title="Time Series - Error",
                width=600,
                height=500
            )
            return fig