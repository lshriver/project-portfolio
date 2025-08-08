## Updated vector field plot function to try for neuron bifurcation app:

```py    
def _add_vector_field_2d(self, fig, params, var_names):
        """Add vector field to 2D phase portrait, using model‐specific default bounds."""

        # 1) model‐specific default ranges
        defaults = {
            'fitzhugh_nagumo': ([-3,   3],   [-3,   3]),
            'hodgkin_huxley': ([-80,  20],  [0.0,  1.0]),
            'morris_lecar':   ([-80,  40],  [0.0,  1.0]),
            'izhikevich':     ([-80,  30],  [-20,  20]),
            'wilson_cowan':   ([0.0,  1.0], [0.0,  1.0]),
            'integrate_fire': ([-80,   0],  [0.0,  10.0]),
        }
        mt = self.neuron_model.model_type
        x_range, y_range = defaults.get(mt, ([-3,3],[-3,3]))

        # 2) optionally extend to cover any plotted trajectories
        if fig.data:
            all_x, all_y = [], []
            for tr in fig.data:
                if hasattr(tr, 'x') and hasattr(tr, 'y') and tr.x is not None and tr.y is not None:
                    all_x += list(tr.x)
                    all_y += list(tr.y)
            if all_x and all_y:
                pad = 0.5
                x_lo = min(x_range[0], min(all_x) - pad)
                x_hi = max(x_range[1], max(all_x) + pad)
                y_lo = min(y_range[0], min(all_y) - pad)
                y_hi = max(y_range[1], max(all_y) + pad)
                x_range, y_range = [x_lo, x_hi], [y_lo, y_hi]

        # 3) build the grid
        x_grid = np.linspace(x_range[0], x_range[1], 15)
        y_grid = np.linspace(y_range[0], y_range[1], 15)
        X, Y = np.meshgrid(x_grid, y_grid)

        # 4) compute the vector field
        DX = np.zeros_like(X); DY = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    deriv = self.neuron_model.equations([X[i,j], Y[i,j]], 0, params)
                    DX[i,j], DY[i,j] = deriv[0], deriv[1]
                except:
                    DX[i,j], DY[i,j] = 0,0

        # 5) normalize and plot arrows
        M = np.hypot(DX, DY)
        M[M==0] = 1
        DX_norm, DY_norm = DX/M, DY/M
        scale = 0.1
        for i in range(0, x_grid.size, 2):
            for j in range(0, y_grid.size, 2):
                fig.add_trace(go.Scatter(
                    x=[X[j,i], X[j,i] + scale * DX_norm[j,i]],
                    y=[Y[j,i], Y[j,i] + scale * DY_norm[j,i]],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
```