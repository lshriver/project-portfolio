import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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
            height = 500
        )

        return fig
    
class TrajectoryPlotter:
    """Handles neural time series trajectory plotting"""

    def __init__(self, neuron_model):
        self.neuron_model = neuron_model
    
    def plot_time_series(self, params, initial_condition, t_max, num_points):
        """"Create time series plot of variables"""
        t = np.linspace(0, t_max, num_points)
        var_names = self.neuron_model.get_variable_names()
        n_vars = len(var_names)

        try:
            # integrate once
            sol = odeint(
                lambda y, tt: self.neuron_model.equations(y, tt, params), 
                initial_condition, t
            )

            # make one row per variable
            fig = make_subplots(
                rows = n_vars, cols = 1,
                shared_xaxes = True,
                vertical_spacing = 0.05,
                subplot_titles = var_names
            )

            colors = ["#4C48F2", "#70e000"]

            for i, var_name in enumerate(var_names):
                fig.add_trace(
                    go.Scatter(
                        x = t,
                        y = sol[:, i],
                        mode = 'lines',
                        line = dict(color=colors[i % len(colors)], width=2),
                        name = var_name
                    ),
                    row = i + 1, col = 1
                )
                # y-axis label per row
                fig.update_yaxes(title_text=var_name, row=i+1, col=1)

            # only the bottom subplot needs an x-axis label
            fig.update_xaxes(title_text="Time", row=n_vars, col=1)

            fig.update_layout(
                title = "Time Series", 
                hovermode = 'x unified',
                showlegend = False,
                width = 600,
                height = 300 * n_vars 
            )

            return fig

        except Exception as e:
            # fallback: single error annotation
            fig = go.Figure()
            fig.add_annotation(
                text = f"Error computing trajectory: {e}",
                xref = "paper", yref = "paper",
                x = 0.5, y = 0.5, showarrow = False,
                font = dict(size=16, color="red")
            )
            fig.update_layout(
                title = "Time Series - Error",
                width = 600,
                height = 300 * max(1, n_vars)
            )
            return fig