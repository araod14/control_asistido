import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def compute_derivatives(y, t, order=3):
    dt = t[1] - t[0]
    dydt = np.gradient(y, dt)
    if order == 2:
        d2ydt2 = np.gradient(dydt, dt)
        return dydt, d2ydt2
    elif order == 3:
        d2ydt2 = np.gradient(dydt, dt)
        d3ydt3 = np.gradient(d2ydt2, dt)
        return dydt, d2ydt2, d3ydt3
    else:
        raise ValueError("Orden no soportado")

def least_squares_fit(y, u, t, order=2):
    if order == 2:
        dy, d2y = compute_derivatives(y, t, order=2)
        X = np.column_stack((d2y, dy, y, u))  # [a2, a1, a0, b0]
    elif order == 3:
        dy, d2y, d3y = compute_derivatives(y, t, order=3)
        X = np.column_stack((d3y, d2y, dy, y, u))  # [a3, a2, a1, a0, b0]
    else:
        raise ValueError("Orden no soportado")

    if order == 2:
        dy, d2y = compute_derivatives(y, t, order=2)
        X = np.column_stack((dy, y, u))  # coef: [a1, a0, b0]
        Y = d2y
    elif order == 3:
        dy, d2y, d3y = compute_derivatives(y, t, order=3)
        X = np.column_stack((d2y, dy, y, u))  # coef: [a2, a1, a0, b0]
        Y = d3y

    theta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    return theta

# Ejemplo de uso
if __name__ == "__main__":
    # Simular sistema real de 2º orden: y'' + 2y' + 5y = 3u
    t = np.linspace(0, 10, 1000)
    u = np.ones_like(t)
    from scipy.signal import lsim, TransferFunction

    sys = TransferFunction([3], [1, 2, 5])
    tout, y, _ = lsim(sys, U=u, T=t)

    y = savgol_filter(y, window_length=51, polyorder=3)

    theta2 = least_squares_fit(y, u, t, order=2)
    print("Coeficientes identificados (2º orden):", theta2)

    theta3 = least_squares_fit(y, u, t, order=3)
    print("Coeficientes identificados (3º orden):", theta3)
