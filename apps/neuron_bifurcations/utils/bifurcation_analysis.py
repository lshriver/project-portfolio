import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import odeint
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')
import importlib.util
import os

colormaps_path = os.path.abspath(
  os.path.join(os.path.dirname(__file__),
               "..", "..", "..", "utils", "colormaps.py")
)
colormaps_spec = importlib.util.spec_from_file_location("parent_utils_colormaps", colormaps_path)
parent_utils_colormaps = importlib.util.module_from_spec(colormaps_spec)
colormaps_spec.loader.exec_module(parent_utils_colormaps)

class BifurcationAnalyzer:
    """Handles neural bifurcation diagram genration and analysis"""

    def __init__(self, neuron_model):
        self.neuron_model = neuron_model

    def plot_bifurcation_diagram(self, params, bifurcation_param, param_min, param_max, num_param_points, initial_condition, t_max):
        """Generate bifurcation diagram by vvaryig a single parameter"""
        param_values = np.linspace(param_min, param_max, num_param_points)
        var_names = self.neuron_model.get_variable_names()

        # Use first variable for bifurcation diagram
        bifurcation_data = []
        param_data = []

        # Time vector - use longer transient time to reach attractor
        t_transient = max(50, t_max)    # Let transients die out
        t_sample = 50   # Sample final protion
        t_total = t_transient + t_sample
        
        t = np.linspace(0, t_total, int(t_total * 50))  # Fine time resolution

        for param_val in param_values:
            try: 
                # Update parameter
                current_params = params.copy()
                current_params[bifurcation_param] = param_val

                # Integrate neural ODE
                sol = odeint(
                    lambda y, t: self.neuron_model.equations(y, t, current_params), initial_condition, t
                )

                # Extract final portion after transients
                transient_idx = int(len(t) * t_transient / t_total)
                final_solution = sol[transient_idx:, 0]     # First variable
                final_time = t[transient_idx:]

                # Find local extrema (peaks and troughts) to identify limit cycles
                extrema = self._find_extrema(final_solution, final_time)

                if len(extrema) > 0:
                    # Limit cycle or periodic behavior
                    for ext_val in extrema:
                        bifurcation_data.append(ext_val)
                        param_data.append(param_val)
                else:
                    # Fixed point behavior
                    final_value = np.mean(final_solution[-100:])    # Average of last points
                    bifurcation_data.append(final_value)
                    param_data.append(param_val)

            except Exception as e:
                # Skip problematic parameter values
                print(f"Error at {bifurcation_param}={param_val}: {e}")
                continue
        
        # Create bifurcation diagram
        fig = go.Figure()   

        fig.add_trace(go.Scatter(
            x = param_data,
            y = bifurcation_data,
            mode = 'markers',
            marker = dict(
                size = 6,
                color = t,
                colorscale = parent_utils_colormaps.BLUE_TO_PURPLE,
                opacity = 0.7
            ),
            name = f'{var_names[0]} extrema',
            showlegend = False
        ))

        fig.update_layout(
            title = f'Bifurcation Diagram - {bifurcation_param} vs. {var_names[0]}',
            xaxis_title = bifurcation_param,
            yaxis_title = f'{var_names[0]}',
            hovermode = 'closest',
            width = 800,
            height = 600,
            paper_bgcolor='rgba(0, 0, 0, 0.5)',
            plot_bgcolor = 'rgba(0, 0, 0, 0.5)',
            margin=dict(l=40, r=40, t=40, b=40)
        )

        return fig
    
    def _find_extrema(self, solution, time, min_prominence=None):
        """Find local extrema in the solutin to identify periodic behavior"""
        if len(solution) < 10:
            return []
        
        # Adaptive prominence based on solution variance
        if min_prominence is None:
            solution_std = np.std(solution)
            min_prominence = max(0.01, solution_std * 0.1)

        # Find peaks
        peaks, peak_props = find_peaks(solution, prominence=min_prominence, distance=5)
        peak_values = solution[peaks] if len(peaks) > 0 else []

        # Find troughts (negative peaks)
        troughs, trough_props = find_peaks(-solution, prominence=min_prominence, distance=5)
        trough_values = solution[troughs] if len(troughs) > 0 else []

        # Combine and filter extrema
        all_extrema = list(peak_values) + list(trough_values)

        # Remove duplicates and outliers
        if len(all_extrema) > 0:
            all_extrema = np.array(all_extrema)
            # Remove outliers beyond 3 standard deviations
            mean_val = np.mean(all_extrema)
            std_val = np.std(all_extrema)
            if std_val > 0:
                mask = np.abs(all_extrema - mean_val) <= 3 * std_val
                all_extrema = all_extrema[mask]

        return all_extrema
    
    def plot_poincare_map(self, params, initial_condition, t_max, plane_coord=2, plane_value=0.0):
        """
        Generate Poincare map for 3D systems
        plane_coord: corrdinate index for the Poincare plan (0, 1, or 2)
        plane_value: value of the plane
        """
        var_names = self.neuron_model.get_variable_names()

        if len(var_names) != 3:
            return None     # Poincare maps only for 3D systems
        
        # Integrate for long time to get good sampling
        t = np.linspace(0, t_max * 2, int(t_max*100))

        try:
            sol = odeint(
                lambda y, t: self.neuron_model.equations(y, t, params),
                initial_condition, t
            )

            # Find intersections with the Poincare maps
            intersections = self._find_plane_intersections(
                sol, plane_coord, plane_value
            )

            if len(intersections) == 0:
                return None
        
            # Create scatter plot of intersecitons
            fig = go.Figure()

            # Determine which coordinates to plot
            coords = [0, 1, 2]
            coords.remove(plane_coord)
            x_coord, y_coord = coords[0], coords[1]

            intersection_array = np.array(intersections)

            fig.add_trace(go.Scatter(
                x = intersection_array[:, x_coord],
                y = intersection_array[:, y_coord],
                mode = 'markers',
                marker = dict(
                    size = 4,
                    color = np.arange(len(intersections)),
                    colorscale = parent_utils_colormaps.BLUE_TO_PURPLE,
                    showscale = True,
                    colorbar = dict(title="Intersection #")
                ),
                name = 'Poincare points'
            ))

            fig.update_layout(
                title = f'Poincare Map - {var_names[plane_coord]} = {plane_value}',
                xaxis_title = var_names[x_coord],
                yaxis_title = var_names[y_coord],
                hovermode = 'closest',
                width = 600,
                height = 600,
                paper_bgcolor='rgba(0, 0, 0, 0.5)',
                plot_bgcolor = 'rgba(0, 0, 0, 0.5)',
                margin=dict(l=40, r=40, t=40, b=40)
            )

            return fig
    
        except Exception as e:
            print(f"Error computing Poincare map: {e}")
            return None

    def _find_plane_intersections(self, solution, plane_coord, plane_value, tolerance=1e-3):
        """Find intersections of t rajectory with a plane"""
        intersections = []

        plane_values = solution[:, plane_coord]

        # Find where trajectory crosses the plane
        for i in range(1, len(solution)):
            prev_val = plane_values[i-1] - plane_value
            curr_val = plane_values[i] - plane_value

            # Check for sign change (crossing)
            if prev_val * curr_val < 0:
                # Linear interpolation to find exact crossing point
                t_cross = abs(prev_val) / (abs(prev_val) + abs(curr_val))
                intersection = (1 - t_cross) * solution[i-1] + t_cross * solution[i]
                intersections.append(intersection)

        return intersections
    
    def analyze_stability(self, params, equilibrium_point, perturbation=1e-6):
        """
        Analyze stability of a neural equilibrium point using numerical linearlization
        """
        var_names = self.neuron_model.get_variable_names()
        n_vars  = len(var_names)

        # Compute Jacobian numerically
        jacobian = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            # Perturb each variable
            perturbed_state_pos = equilibrium_point.copy()
            perturbed_state_neg = equilibrium_point.copy()
            perturbed_state_pos[i] += perturbation
            perturbed_state_neg[i] -= perturbation
            
            # Compute derivatives
            derivs_pos = self.neuron_model.equations(perturbed_state_pos, 0, params)
            derivs_neg = self.neuron_model.equations(perturbed_state_neg, 0, params)

            # Central difference
            jacobian[:, i] = (np.array(derivs_pos) - np.array(derivs_neg)) / (2 * perturbation)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(jacobian)

        # Classify stability
        real_parts = np.real(eigenvalues)

        if np.all(real_parts < 0):
            stability = "Stable (sink)"
        elif np.all(real_parts > 0):
            stability = "Unstable (source)"
        elif np.any(real_parts > 0) and np.any(real_parts < 0):
            stability = "Saddle point"
        else: 
            stability = "Marginally stable"

        return {
            'jacobian': jacobian,
            'eigenvalues': eigenvalues,
            'stability': stability
        }