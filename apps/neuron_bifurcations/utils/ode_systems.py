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
            'a': (0.7, (0.1, 2.0), 'Recovery variable parameter $a$'),
            'b': (0.8, (0.1, 2.0), 'Recovery variable parameter $b$'),
            'tau': (12.5, (1.0, 50.0), 'Recovery time constant $\tau$'),
            'J': (0.5, (-2.0, 3.0), 'Applied Current $J$')
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
            'g_Na': (120.0, (50.0, 200.0), '$g_{Na} \ \mathrm{(mS/cm^2)}$ - Sodium conductance'),
            'g_K': (36.0, (10.0, 80.0), '$g_K \ \mathrm{(mS/cm^2)}$ - Potassium conductance'),
            'g_L': (0.3, (0.1, 1.0), '$g_L \ \mathrm{(mS/cm^2)}$ - Leak conductance'),
            'E_Na': (50.0, (40.0, 60.0), '$E_{Na} \ \mathrm{(mV)}$ Sodium reversal potential'),
            'E_K': (-77.0, (-90.0, -60.0), '$E_{K} \ \mathrm{(mV)}$ Potassium reversal potential'),
            'E_L': (-54.4, (-70.0, -40.0), '$E_{L} \ \mathrm{(mV)}$ Leak reversal potential'),
            'I': (10.0, (-50.0, 100.0), '$I \ \mathrm{(\mu A/cm^2)}$ - Applied current'),
            'C': (1.0, (0.5, 2.0), '$C \ \mathrm{(\mu F/cm^2)}$ - Membrane capacitance')
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
        self.parameters = {
            'C': (20.0, (1.0, 100.0), '$C \\ \mathrm{(\\mu F/cm^2)}$ - Membrane capacitance'),
            'g_Ca': (4.4, (1.0, 10.0), '$g_{Ca} \ \mathrm{(mS/cm^2)}$ - Calcium conductance'),
            'g_K': (8.0, (2.0, 20.0), '$g_K \ \mathrm{(mS/cm^2)}$ - Potassium conductance'),
            'g_L': (2.0, (0.5, 5.0), '$g_L \ \mathrm{(mS/cm^2)}$ - Leakage conductance'),
            'E_Ca': (120.0, (100.0, 140.0), '$E_{Ca} \ \mathrm{(mV)}$ - Calcium reversal potential'),
            'E_K': (-84.0, (-100.0, -70.0), '$E_K \ \mathrm{(mV)}$ - Potassium reversal potential'),
            'E_L': (-60.0, (-80.0, -40.0), '$E_L \ \mathrm{(mV)}$ - Leak reversal potential'),
            'phi': (0.04, (0.01, 0.1), '$\phi$ - Temperature factor'),
            'V1': (-1.2, (-10.0, 10.0), '$V_1 \ \mathrm{(mV)}$ - Half-activation voltage'),
            'V2': (18.0, (10.0, 30.0), '$V_2 \ \mathrm{(mV)}$ - Activation slope'),
            'V3': (2.0, (-10.0, 20.0), '$V_3 \ \mathrm{(mV)}$ - Half-inactivation voltage'),
            'V4': (30.0, (15.0, 50.0), '$V_4 \ \mathrm{(mV)}$ - Inactivation slope'),
            'I': (40.0, (-100.0, 200.0), '$I \ \mathrm{(\mu A/cm^2)}$ - Applied current')
        }
        self.variable_names = ['V', 'W']
        self.description = """
        - Morris-Lecar model describes voltage-gated calcium and potassium dynamics. 
        - $V$ is membrane potential, $W$ is potassium channel activation.
        - Exibits oscillatory behavior.
        """
        self.equations_latex = [
            r"C\frac{dV}{dt} = I - g_{Ca}\,m_{\infty}(V)\,(V- E_{Ca})"
            r" - g_K\,W\,(V-E_K) - g_L\,(V-E_L)",
            r"\frac{dW}{dt} = \frac{w_{\infty}(V) - W}{\tau_w(V)}"
        ]   

    def setup_izhikevich(self):
        """Izhikevich neuron model: efficient spiking model"""
        self.parameters = {
            'a': (0.02, (0.001, 0.2), '$a$ - Recovery time scale'),
            'b': (0.2, (0.01, 0.5), '$b$ - Recovery sensitivity'),
            'c': (-65.0, (-80.0, -40.0), '$c \ \mathrm{(mV)}$ - Reset potential'),
            'd': (6.0, (0.1, 20.0), '$d$ - Recovery increment'),
            'I': (14.0, (-20.0, 50.0), '$I \ \mathrm{(pA)}$ - Applied current')
        }
        self.variable_names = ['v', 'u']
        self.description = """
        - Izhikevich model combines efficiency with bilogical realism.
        - $v$ is the membrane potential, $u$ is the recovery variable.
        - Can reproduce various neuronal firing patterns.
        """
        self.equations_latex = [
            r"\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I",
            r"\frac{du}{dt} = a(bv - u)"
        ]

    def setup_wilson_cowan(self):
        """Wilson-Cowan neural field model: excitatory and inhibitory populations"""
        self.parameters = {
            'tau_E': (1.0, (0.1, 10.0), '$ \\tau_{E}$ - Excitatory time constant'),
            'tau_I': (1.0, (0.1, 10.0), '$ \\tau_{I}$ - Inhibitory time constant'),
            'c_EE': (16.0, (5.0, 30.0), '$c_{EE}$ - Excitatory-Excitatory coupling'),
            'c_EI': (12.0, (5.0, 25.0), '$c_{EI}$ - Excitatory-Inhibitory coupling'),
            'c_IE': (15.0, (5.0, 30.0), '$c_{IE}$ - Inhibitory-Excitatory coupling'),
            'c_II': (3.0, (1.0, 15.0), '$c_{II}$ - Inhibitory-Inhibitory coupling'),
            'a_E': (1.3, (0.5, 3.0), '$a_E$ - Excitatory gain'),
            'a_I': (2.0, (0.5, 4.0), '$a_I$ - Inhibitory gain'),
            'theta_E': (4.0, (1.0, 10.0), '$\theta_E$ - Excitatory threshold'),
            'theta_I': (3.7, (1.0, 10.0), '$\theta_I$ - Inhibitory threshold'),
            'P': (1.0, (-5.0, 15.0), '$P$ - External input') 
        }
        self.variable_names = ['E', 'I']
        self.description = """
        - Wilson-Cowan model describes dynamics of excitatory ($E$) and inhibitory ($I$) neural populations.
        - Shows oscillations, bistability, and various activity patterns.
        - Imortant for understanding neural field dynamics.
        """
        self.equations_latex = [
            r"\frac{dE}{dt} = \frac{1}{\tau_E}[-E + S_E(c_{EE}E - c_{EI}I + P)]",
            r"\frac{dI}{dt} = \frac{1}{\tau_I}[-I + S_I(c_{IE}E - c_{II}I)]"
        ]

    def setup_integrate_fire(self):
        """Leaky Integrate-and-Fire neuron with adaptation"""
        self.parameters = {
            'tau_m': (20.0, (5.0, 100.0), '$\tau_m \ \mathrm{(ms)}$ - Membrane time constant'),
            'tau_w': (144.0, (50.0, 500.0), '$\tau_w \ \mathrm{(ms)}$ - Adaptation time constant'),
            'a': (4.0, (0.0, 20.0), '$a \ \mathrm{(nS)}$ - Subthreshold adaptation'),
            'b': (0.0805, (0.0, 0.5), '$b \ \mathrm{(nA)}$ - Spike-triggered adaptation'),
            'E_L': (-70.6, (-80.0, -60.0), '$E_L \ \mathrm{(mV)}$ - Leak reversal potential'),
            'Delta_T': (2.0, (0.5, 10.0), '$\Delta_T \ \mathrm{(mV)}$ - Slope factor'),
            'V_T': (-50.4, (-60.0, -40.0), '$V_T \ \mathrm{(mV)}$ - Threshold potential'),
            'I': (0.65, (-2.0, 5.0), '$I \ \mathrm{(nA)}$ - Applied current')
        }
        self.variable_names = ['V', 'w']
        self.description = """
        - Adaptive Exponential Integrate-and-Fire model.
        - $V$ is membrane potential, $w$ is adaptation current.
        - Shows spike-freqauency adaptation and various firing patterns.
        """
        self.equations_latex = [
            r"\frac{dV}{dt} = \frac{1}{\tau_m}[-(V - E_L) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} - w + I]",
            r"\frac{dw}{dt} = \frac{1}{\tau_w}[a(V - E_L) - w]"
        ]

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
        
        elif self.model_type == 'morris_lecar':
            V, W = state
            C = params['C']
            g_Ca, g_K, g_L = params['g_Ca'], params['g_K'], params['g_L']
            E_Ca, E_K, E_L = params['E_Ca'], params['E_K'], params['E_L']
            phi, V1, V2, V3, V4 = params['phi'], params['V1'], params['V2'], params['V3'], params['V4']
            I_app = params['I']  # could be a function of t

            # Steady-state curves
            m_inf = 0.5 * (1 + np.tanh((V - V1) / V2))
            w_inf = 0.5 * (1 + np.tanh((V - V3) / V4))
            # Time constant
            tau_w = 1 / (phi * np.cosh((V - V3) / (2 * V4)))

            # Ionic currents 
            I_Ca = g_Ca * m_inf * (V - E_Ca)
            I_K = g_K * W * (V - E_K)
            I_L = g_L * (V - E_L)

            dVdt = (I_app - I_Ca - I_K - I_L) / C
            dWdt = (w_inf - W) / tau_w

            return [dVdt, dWdt]
        
        elif self.model_type == 'izhikevich':
            v, u = state
            a, b, I = params['a'], params['b'], params['I']

            return [
                0.04 * v**2 + 5*v + 140 - u + I,
                a * (b*v - u)
            ]
        
        elif self.model_type == 'wilson_cowan':
            E, I_pop = state # Renamed I to I_pop to avoid conflic with current
            tau_E, tau_I = params['tau_E'], params['tau_I']
            c_EE, c_EI, c_IE, c_II = params['c_EE'], params['c_EI'], params['c_IE'], params['c_II']
            a_E, a_I = params['a_E'], params['a_I']
            theta_E, theta_I = params['theta_E'], params['theta_I']
            P = params['P']

            # Sigmoid activation function
            def S_E(x):
                return 1 / (1 + np.exp(-a_E * (x - theta_E)))
            
            def S_I(x):
                return 1 / (1 + np.exp(-a_I * (x - theta_I)))
            
            return [
                (-E + S_E(c_EE * E - c_EI * I_pop + P)) / tau_E,
                (-I_pop + S_I(c_IE * E - c_II * I_pop)) / tau_I
            ]
        
        elif self.model_type == 'integrate_fire':
            V, w = state
            tau_m, tau_w = params['tau_m'], params['tau_w']
            a, b = params['a'], params['b']
            E_L, Delta_T, V_T = params['E_L'], params['Delta_T'], params['V_T']
            I = params['I']

            # Exponential term with numerical stability
            exp_term = Delta_T * np.exp(min((V - V_T) / Delta_T, 20))   # Cap to prevent overflow

            return [
                (-(V - E_L) + exp_term - w + I) / tau_m,
                (a * (V - E_L) - w) / tau_w
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
