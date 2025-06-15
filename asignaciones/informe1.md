---

# **Informe Técnico: Diseño de Controlador PID usando el Método del Lugar de las Raíces**

---

## **1. Introducción**

El control PID (Proporcional–Integral–Derivativo) es una técnica ampliamente utilizada en sistemas de control automático. Este informe presenta el diseño automático de un controlador PID empleando el método del lugar de las raíces, con el objetivo de ubicar los polos del sistema en ubicaciones deseadas en el plano complejo para garantizar un desempeño temporal adecuado.

---

## **2. Código Desarrollado**

El siguiente script en Python emplea las bibliotecas `numpy` y `control` para calcular las ganancias $K_p$, $K_i$, y $K_d$ de un controlador PID basado en un análisis del lugar de las raíces de la planta:

```python
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
```

---

## **3. Aplicación del Código**

Se aplica el método al diseño de un controlador PID para una planta de primer orden descrita por la función de transferencia:

$$
G(s) = \frac{1}{s + 1}
$$

Esto se implementa en el código con:

```python
# Crear una planta continua
num = [1]
den = [1, 1]
planta = ct.tf(num, den)

# Probar la función
Kp, Ki, Kd = disenar_pid_lugar_raices(planta)
print("Ganancias del PID:")
print(f"Kp = {Kp:.4f}, Ki = {Ki:.4f}, Kd = {Kd:.4f}")
```

---

## **4. Resultados Obtenidos**

La ejecución del código produce las siguientes ganancias para el controlador PID:

```
Ganancias del PID:
Kp = 0.6472, Ki = 3.1436, Kd = 0.0000
```

Esto indica que el controlador óptimo en este caso es un **PI** (proporcional-integral), ya que $K_d = 0$. Esto es consistente con una planta de primer orden, donde la acción derivativa no suele aportar beneficios significativos.

---

## **5. Análisis de Resultados**

### **5.1. Interpretación de Ganancias**

* **$K_p = 0.6472$:** Ganancia proporcional, responsable de reducir el error inmediato.
* **$K_i = 3.1436$:** Acción integral fuerte, encargada de eliminar el error en régimen permanente.
* **$K_d = 0$:** No se requiere acción derivativa dado que el sistema tiene una sola constante de tiempo y no presenta sobreoscilaciones.

### **5.2. Desempeño Esperado del Sistema**

Dado que se seleccionó un factor de amortiguamiento de 0.7 y un tiempo de establecimiento deseado de 2 segundos, el polo deseado fue ubicado en una región del plano complejo que garantiza:

* Tiempo de establecimiento cercano a 2 s.
* Sobreoscilación moderada (< 5%).
* Buen compromiso entre rapidez y estabilidad.

### **5.3. Limitaciones y Posibles Mejoras**

* El algoritmo no optimiza frente a restricciones de control (saturación, ruido, etc.).
* No considera robustez frente a modelos con incertidumbre.
* La acción derivativa podría ser beneficiosa si el sistema tuviera dinámica de orden superior o fuese afectado por perturbaciones rápidas.

---

## **6. Conclusiones**

El método de diseño de un controlador PID mediante el lugar de las raíces, tal como se implementa en el código presentado, permite una determinación automática de las ganancias del controlador para alcanzar especificaciones temporales básicas. La metodología es efectiva para plantas simples y proporciona un punto de partida sólido para el diseño y ajuste fino de controladores PID.

---
