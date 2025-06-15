import numpy as np
import control as ct
import matplotlib.pyplot as plt

def disenar_pid_bode(planta, margen_fase_deseado=60, wc_deseado=None):
    mag, phase, omega = ct.bode_plot(planta, dB=True, plot=True)
    if planta.dt is not None and planta.dt > 0:
        s = ct.TransferFunction([1, 0], [1], dt=planta.dt)
    else:
        s = ct.TransferFunction([1, 0], [1])

    K_planta = abs(ct.dcgain(planta))

    omega_eval = np.logspace(-2, 2, 1000)
    response = ct.frequency_response(planta, omega_eval)
    mag = np.abs(response)

    if wc_deseado is None:
        indices = np.where(mag >= 1.0)[0]
        if len(indices) > 0:
            wc_deseado = omega_eval[indices[0]]
        else:
            wc_deseado = omega_eval[-1]

    Kp = 1.0 / K_planta
    Ki = 0.1 * Kp * wc_deseado
    Kd = 0.1 * Kp / wc_deseado

    return Kp, Ki, Kd

# Crear una planta de ejemplo: G(s) = 1 / (s^2 + 3s + 2)
num = [1]
den = [1, 3, 2]
planta = ct.TransferFunction(num, den)

# Llamar a la función
Kp, Ki, Kd = disenar_pid_bode(planta)

# Mostrar resultados
print(f"Kp: {Kp:.4f}, Ki: {Ki:.4f}, Kd: {Kd:.4f}")

# Mostrar el diagrama de Bode
plt.show()
