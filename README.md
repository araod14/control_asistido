# Sistema de Identificación y Control PID

## Descripción

Esta aplicación proporciona una interfaz gráfica para la identificación de modelos de sistemas dinámicos y el diseño de controladores PID. Permite cargar datos experimentales, identificar diferentes tipos de modelos (lineales y no lineales), y diseñar controladores PID utilizando diversos métodos.

## Características principales

- **Identificación de modelos**:
  - Modelos de segundo orden
  - Modelos de segundo orden con retardo
  - Modelos lineales ARX
  - Modelos no lineales ARMAX
  - Modelos no lineales OE (Output Error)

- **Diseño de controladores PID**:
  - Método Ziegler-Nichols
  - Lugar de las raíces
  - Respuesta en frecuencia (Bode)
  - Aprendizaje automático con Scikit-Learn
  - Configuración manual

- **Visualización**:
  - Gráficos comparativos de modelos identificados vs datos reales
  - Respuesta del sistema en lazo cerrado con el controlador PID
  - Diagramas de Bode y lugar de las raíces

- **Funcionalidades adicionales**:
  - Registro detallado de eventos
  - Exportación de resultados a PDF
  - Análisis de estabilidad y márgenes de ganancia/fase

## Requisitos del sistema

- Python 3.7 o superior
- Bibliotecas requeridas:
  - numpy
  - matplotlib
  - control
  - scikit-learn
  - tkinter

## Instalación

1. Clonar el repositorio o descargar el archivo Python
2. Instalar las dependencias:
   ```
   pip install numpy matplotlib control scikit-learn
   ```

## Uso

1. Ejecutar el script:
   ```
   python nombre_del_archivo.py
   ```
2. Cargar un archivo de datos con el formato:
   - Primera columna: tiempo
   - Segunda columna: entrada
   - Tercera columna: salida
3. Seleccionar el tipo de modelo a identificar
4. Elegir el método de diseño PID
5. Analizar los resultados en las gráficas y el registro de eventos

## Formato del archivo de datos

El archivo de entrada debe ser un archivo de texto con tres columnas separadas por espacios o tabulaciones:
```
tiempo1 entrada1 salida1
tiempo2 entrada2 salida2
...
```

## Ejemplo de uso

1. Cargar datos experimentales de un sistema
2. Identificar un modelo de segundo orden con retardo
3. Diseñar un controlador PID usando Ziegler-Nichols
4. Analizar la respuesta en lazo cerrado y los márgenes de estabilidad
5. Exportar los resultados a PDF

## Licencia

Este proyecto está bajo la licencia MIT.

## Contacto

Para preguntas o sugerencias, por favor contacte al desarrollador.
