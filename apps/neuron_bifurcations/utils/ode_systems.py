import numpy as np
import sympy as sp

class NeuronModel:
    """Base class for neruonal signaling models with common functionality"""

    def __init__(self, model_type):
        self.model_type = model_type
        self.setup_model()

    def setup_model(self):
        """Setup the specific neuronal model based on type"""
        if self.model_type == 'fitzhugh_nagumo':
            self.setup_fitzhugh_nagumo()
        elif self.model_type == 'hodgkin_huxley':
            self.setup_hodgkin_huxley()
        elif self.model_type == 'morris_lecar':
            self.setup_morris_lecar()
        elif self.model_type == 'izhikevich':
            self.setup_izhikevich()
        elif self.model_type == 'wilson_cowan':
            self.setup_wilson_cowan()
        elif self.model_type == 'integrate_fire':
            self.setup_integrate_fire()
        else:
            raise ValueError(f"Unknown neuron model type: {self.model_type}")
        
    def setup_fitzhugh_nagumo(self):
        """FitzHugh-Nagumo neruon model: Simplified Hodgkin-Huxley"""
        self.parameters = {
            'a': (0.7, (0.1, 2.0), 'Recovery variable parameter'),
            'b': (0.8, (0.1, 2.0), 'Recovery variable parameter'),
            'tau': (12.5, (1.0, 50.0), 'Recovery time constant'),
            'J': (0.5, (-2.0, 3.0), 'Applied Current')
        }
        self.variable_names = ['V', 'W']
        self.description = """
        - FitzHugh-Nagumo model is a simplified version of the HH model. 
        - V denotes the membrane potential and W denotes denotes the recovery variable. 
        - Shows excitable dynamics and can exhibit spiking behavior.
        """
        self.equations_latex = [
            r"\frac{dV}{dt} = V - \frac{V^3}{3} - W + J",
            r"\frac{dW}{dt} = \frac{1}{\tau}(V + a - bW)"
        ]

    def setup_hodgkin_huxley(self):
        """Simplified Hodgkin-Huxley model (2D reduction)"""
        self.parameters = {
            'g_Na': (120.0, (50.0, 200.0), 'Sodium conductance $g_Na \mathrm{(mS/cm^2)}$'),
            'g_K': (36.0, (10.0, 80.0), 'Potassium conductance $g_K \mathrm{(mS/cm^2)}$'),
            'g_L': (0.3, (0.1, 1.0), 'Leak conductance $g_L \mathrm{(mS/cm^2)}'),
            'E_Na': (50.0, (40.0, 60.0), 'Sodium reversal potential $E_{Na} \mathrm{(mV)}'),
            'E_K': (-77.0, (-90.0, -60.0), 'Potassium reversal potential $E_{K} \mathrm{(mV)}'),
            'E_L': (-54.4, (-70.0, -40.0), 'Leak reversal potential $E_{L} \mathrm{(mV)}'),
            'I': (10.0, (-50.0, 100.0), 'Applied current $I \mathrm{(\mu A/cm^2)}$'),
            'C': (1.0, (0.5, 2.0), 'Membrane capacitance $C \mathrm{(\mu F/cm^2)}$')
        }
        self.variable_names = ['V', 'n']
        self.description = """
        - Simplified Hodgkin-Huxley mdoel with voltage $V$ and potassium gating variable $n$.
        - Classic model for action potential generation in neruons.
        - Shows excitablity threshoold and spike generation.
        """
        self.equations_latex = [
            r"\frac{dV}{dt} = \frac{1}{C}[I - g_{Na}m_{\infty}^3(V)(V-E_{Na}) - g_K n^4(V-E_K) - g_L(V-E_L)]",
            r"\frac{dn}{dt} = \frac{n_{\infty}(V) - n}{\tau_n (V)}"
        ]

    def setup_morris_lecar(self):
        """Morris-Lecar neuron model: voltage-gated calcium and potassium"""

    def setup_izhikevich(self):
        """Nzhikevich neuron model: efficient spiking model"""

    def setup_wilson_cowan(self):
        """Wilson-Cowan neural field model: excitatory and inhibitory populations"""

    def setup_integrate_fire(self):
        """Leaky Integrate-and-Fire neuron with adaptation"""

    def equations(self, state, t, params):
        """
        Define the neuronal model ODEs
            - state: current state vector
            - t: time 
            - params: dictionary of parameters
        """
        if self.model_type == 'fitzhugh_nagumo':
            V, W = state
            a, b, tau, J = params['a'], params['b'], params['tau'], params['J']
            return [
                V - V**3/3 - W + J,
                (V + a - b*W) / tau
            ]
        elif self.model_type == 'hodgkin_huxley':
            V, n = state
            g_Na, g_K, g_L = params['g_Na'], params['g_K'], params['g_L']
            E_Na, E_K, E_L = params['E_Na'], params['E_K'], params['E_L']
            I, C = params['I'], params['C']

            # Gating functions
            alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40)/10))
            beta_m = 4 * np.exp(-(V + 65)/18)
            m_inf = alpha_m / (alpha_m + beta_m)
            alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55)/10))
            beta_n = 0.125 * np.exp(-(V + 65)/80)

            # Currents
            I_Na = g_Na * m_inf**3 * (0.8 - n) * (V - E_Na)   # Simplifed h \approx 0.8 - n
            I_K = g_K * n**4 * (V - E_K)
            I_L = g_L * (V - E_L)

            return [
                (I - I_Na - I_K - I_L) / C,
                alpha_n * (1 - n) - beta_n *n
            ]
        
        else:
            raise ValueError(f"Unknown neruon model type: {self.model_type}")
        
    def get_parameters(self):
        """Return parameter definitions"""
        return self.parameters
    
    def get_variable_names(self):
        """Return variable names"""
        return self.variable_names
    
    def get_description(self):
        """Return system description"""
        return self.description
    
    def get_equations_latex(self):
        """Return LaTeX representation of equations"""
        return self.equations_latex
    
    def get_equilibrium_points(self, params):
        """Calculate equilibrium points analytically where possible"""
        if self.model_type == 'fitzhugh_nagumo':
            a, b, tau, I = params['a'], params['b'], params['tau'], params['J']
            return []
        
        elif self.model_type == 'wilson_cowan':
            # Wilson-Cowan equlibria depend on sigmoid funcitons
            # Generally requires numerical methods
            return []
        
        elif self.model_type == 'integrate_fire':
            # Leaky integrate-and-fire typically has one stable equilibria
            E_L, I, a = params['E_L'], params['I'], params['a']
            if a > 0:
                V_eq = E_L + I/a    # Approximate equilibrium
                w_eq = I
                return [[V_eq, w_eq]]
            return []
        else:
            return []   # Most neruon models require numerical methods for equlibria
