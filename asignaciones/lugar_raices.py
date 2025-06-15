import numpy as np
import control as ct

def disenar_pid_lugar_raices(planta, factor_amortiguamiento=0.7, margen_fase=60):
    if planta.dt is not None and planta.dt > 0:
        planta_continua = planta
    else:
        planta_continua = planta

    polos = ct.poles(planta_continua)
    ceros = ct.zeros(planta_continua)

    ts_deseado = 2.0
    wn_deseado = 4 / (factor_amortiguamiento * ts_deseado)

    if planta.dt is not None and planta.dt > 0:
        T = planta.dt
        wn_deseado = min(wn_deseado, np.pi / (4 * T))

    if planta.dt is not None and planta.dt > 0:
        s_continuo = -factor_amortiguamiento * wn_deseado + 1j * wn_deseado * np.sqrt(1 - factor_amortiguamiento ** 2)
        polo_deseado = np.exp(s_continuo * planta.dt)
    else:
        polo_deseado = -factor_amortiguamiento * wn_deseado + 1j * wn_deseado * np.sqrt(1 - factor_amortiguamiento ** 2)

    if len(polos) == 1:
        polo_planta = polos[0]

        if planta.dt is not None and planta.dt > 0:
            K_planta = abs(ct.dcgain(planta))
            Kp = 0.5 / K_planta
            Ki = Kp / (10 * planta.dt)
            Kd = Kp * planta.dt / 10
        else:
            zc = -wn_deseado * (1 + factor_amortiguamiento)

            s = polo_deseado
            gc_s = (s - zc)
            try:
                if hasattr(planta_continua, 'evalfr'):
                    gp_s = planta_continua.evalfr(s)[0][0]
                else:
                    gp_s = ct.evalfr(planta_continua, s)
                    if hasattr(gp_s, '__getitem__'):
                        gp_s = gp_s[0][0]
            except:
                gp_s = ct.evalfr(planta_continua, s)

            Kp = 1 / abs(gc_s * gp_s)
            Ki = Kp * abs(zc)
            Kd = 0

    else:
        if len(polos) >= 2:
            polos_ordenados = sorted(polos, key=lambda x: abs(x.real))
            polo1, polo2 = polos_ordenados[:2]

            if np.iscomplex(polo1):
                wn_actual = abs(polo1)
                zeta_actual = -polo1.real / wn_actual
            else:
                wn_actual = abs(polo1)
                zeta_actual = 1.0

        K_planta = abs(ct.dcgain(planta_continua))

        Kp = wn_deseado ** 2 / (K_planta * abs(polos[0]) ** 2)
        Ki = wn_deseado ** 3 / (K_planta * abs(polos[0]) ** 2)
        Kd = 2 * factor_amortiguamiento * wn_deseado / (K_planta * abs(polos[0]))

    Kp = abs(Kp)
    Ki = abs(Ki)
    Kd = abs(Kd)

    return Kp, Ki, Kd

# -----------------
# PRUEBA DEL MÉTODO
# -----------------
# Crear una planta continua
num = [1]
den = [1, 1]
planta = ct.tf(num, den)

# Probar la función
Kp, Ki, Kd = disenar_pid_lugar_raices(planta)
print("Ganancias del PID:")
print(f"Kp = {Kp:.4f}, Ki = {Ki:.4f}, Kd = {Kd:.4f}")
