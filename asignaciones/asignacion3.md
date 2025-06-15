
---

## **INFORME DE IDENTIFICACIÓN DE SISTEMAS DINÁMICOS USANDO MÍNIMOS CUADRADOS**

### **1. Código Desarrollado**

El objetivo del código es identificar los coeficientes de una ecuación diferencial que modela un sistema dinámico a partir de su respuesta temporal. El método usado es el de **mínimos cuadrados lineales** aplicados sobre derivadas numéricas de la salida simulada.

#### Código fuente:

```python
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
        X = np.column_stack((dy, y, u))  # [a1, a0, b0]
        Y = d2y
    elif order == 3:
        dy, d2y, d3y = compute_derivatives(y, t, order=3)
        X = np.column_stack((d2y, dy, y, u))  # [a2, a1, a0, b0]
        Y = d3y
    else:
        raise ValueError("Orden no soportado")

    theta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    return theta

# Ejemplo de uso
if __name__ == "__main__":
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
```

---

### **2. Aplicación del Código**

Se simula un sistema físico de **segundo orden** descrito por la ecuación diferencial:

$$
y'' + 2y' + 5y = 3u
$$

Esto equivale a una función de transferencia:

$$
G(s) = \frac{3}{s^2 + 2s + 5}
$$

Se aplicó una entrada escalón unitario y se obtuvo la respuesta del sistema con `lsim`. Posteriormente, la señal se suaviza con un **filtro de Savitzky-Golay**, y se usan derivadas numéricas para aplicar el ajuste por mínimos cuadrados tanto para modelos de **2º orden** como de **3º orden**.

---

### **3. Resultados Obtenidos**

La ejecución del código produjo los siguientes resultados:

```
Coeficientes identificados (2º orden): [-1.92972039 -4.82814264  2.89416799]
Coeficientes identificados (3º orden): [ -58.8215307  -119.55889894 -286.37796172  171.86334516]
```

#### Interpretación de los coeficientes:

##### Modelo de 2º orden:

Los coeficientes estimados corresponden a la siguiente ecuación:

$$
\ddot{y} = -1.93\dot{y} -4.83y + 2.89u
$$

Este resultado es cercano al modelo real:

$$
\ddot{y} = -2\dot{y} -5y + 3u
$$

> **Error porcentual estimado:**

* Error en $a_1$ (coeficiente de $\dot{y}$): \~3.5%
* Error en $a_0$ (coeficiente de $y$): \~3.4%
* Error en $b_0$: \~3.7%

##### Modelo de 3º orden:

El sistema original **no** tiene dinámica de tercer orden, por lo que al forzar el ajuste con un modelo más complejo se obtienen:

$$
\dddot{y} = -58.82\ddot{y} -119.56\dot{y} -286.38y + 171.86u
$$

Estos coeficientes **no representan** una dinámica física realista en este caso y son simplemente un sobreajuste a los datos simulados.

---

### **4. Análisis de los Resultados**

#### **a) Precisión del ajuste de segundo orden**

* El modelo de segundo orden identificó adecuadamente los parámetros del sistema real.
* La precisión es alta, con errores menores al 5%, lo que valida la eficacia del método con un preprocesamiento adecuado (suavizado + derivación).

#### **b) Sobreajuste en el modelo de tercer orden**

* El modelo de orden superior muestra inestabilidad en los coeficientes, reflejando un **sobreajuste** a los datos suavizados y las derivadas.
* No es recomendable usar modelos de orden superior al real sin justificación.

#### **c) Importancia del filtrado**

* El uso del filtro de Savitzky-Golay fue clave para evitar amplificación del ruido en las derivadas.
* En datos reales, el filtrado es **crítico** para la estabilidad del método.

#### **d) Aplicabilidad**

Este método es útil en:

* Identificación experimental de sistemas físicos.
* Verificación de modelos dinámicos.
* Sistemas lineales LTI con entrada conocida y salida medida.

---

### **5. Conclusión**

El método de identificación por **mínimos cuadrados sobre derivadas numéricas** ha demostrado ser efectivo para recuperar modelos dinámicos simples como el de segundo orden utilizado. Se ha comprobado que:

* El ajuste es fiable si el orden del modelo coincide con el sistema real.
* Modelos más complejos pueden inducir errores grandes por sobreajuste.
* Es fundamental el preprocesamiento de señales para garantizar precisión.

---

