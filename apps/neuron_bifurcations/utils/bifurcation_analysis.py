import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import odeint
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class BifurcationAnalyzer:
    """Handles neural bifurcation diagram genration and analysis"""

    def __init__(self, neuron_model):
        self.neuron_model = neuron_model