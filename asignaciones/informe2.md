
---

## **INFORME DE DISEÑO DE CONTROLADOR PID UTILIZANDO EL DIAGRAMA DE BODE**

### **1. Código Desarrollado**

```python
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
```

---

### **2. Aplicación del Código**

Se ha diseñado un controlador PID automático a partir del análisis en frecuencia de una planta LTI. La planta empleada como caso de estudio es:

$$
G(s) = \frac{1}{s^2 + 3s + 2}
$$

Esta planta corresponde a un sistema de segundo orden subamortiguado. La función `disenar_pid_bode` calcula las ganancias PID a partir de la ganancia estática del sistema y del cruce por unidad de ganancia del diagrama de Bode.

---

### **3. Resultados Obtenidos**

La ejecución del código genera la siguiente salida:

```
/home/danel149/control_asistido_pc/venv/lib/python3.10/site-packages/control/freqplot.py:435: FutureWarning: bode_plot() return value of mag, phase, omega is deprecated; use frequency_response()
Kp: 2.0000, Ki: 0.0020, Kd: 19.8165
```

Se obtuvo lo siguiente para el controlador PID:

* **Kp (proporcional):** 2.0000
* **Ki (integral):** 0.0020
* **Kd (derivativo):** 19.8165

Además, se muestra gráficamente el **diagrama de Bode** del sistema, lo que permite verificar visualmente el comportamiento en frecuencia, incluyendo márgenes de ganancia y fase.

---

### **4. Análisis de los Resultados**

#### **a) Evaluación de Ganancias**

* El valor **Kp = 2.0** sugiere una ganancia proporcional inversa a la ganancia DC de la planta, normalizando la respuesta del sistema.
* **Ki** tiene un valor bajo, lo que implica una acción integrativa lenta. Esto es adecuado si se busca evitar una acumulación rápida de error y mantener el sistema estable.
* **Kd** es alto, indicando una acción derivativa fuerte, lo cual puede ayudar a mejorar la respuesta transitoria (reduciendo el sobreimpulso y mejorando el amortiguamiento).

#### **b) Comportamiento en Frecuencia**

* El diseño se basa en una estimación de la frecuencia de cruce por unidad de ganancia, lo cual proporciona una forma rápida, aunque aproximada, de sintonizar el PID.
* El margen de fase deseado es de **60 grados**, aunque en este diseño no se afina exactamente para ese margen, sino que se asume una estructura fija para calcular $K_i$ y $K_d$ a partir de $K_p$ y $\omega_c$.

#### **c) Observaciones Técnicas**

* El código genera una advertencia de deprecación en la función `bode_plot()`, lo cual sugiere utilizar `frequency_response()` directamente para evitar posibles problemas en futuras versiones de la biblioteca.
* La elección de $K_i$ y $K_d$ como proporcionales a $K_p \omega_c$ y $K_p / \omega_c$ respectivamente es un enfoque heurístico y puede requerir ajustes finos adicionales.

---

### **5. Conclusión**

Este trabajo presenta una aproximación automática para el diseño de un controlador PID mediante el análisis del diagrama de Bode. Aunque no se trata de una sintonización óptima o robusta, ofrece una forma inicial eficiente de obtener parámetros razonables para comenzar el ajuste fino.

El código puede ser extendido con:

* Análisis de respuesta temporal (escalón, rampa).
* Verificación de estabilidad en lazo cerrado.
* Ajuste automático para cumplir márgenes de fase y ganancia deseados.

---

