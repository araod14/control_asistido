import numpy as np
import matplotlib.pyplot as plt
import control as ct
import logging
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_pdf import PdfPages

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('control_pid.log'),
                        logging.StreamHandler()
                    ])

class PIDControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Identificación y Control PID")
        self.root.geometry("1200x900")
        
        # Variables de control
        self.t = None
        self.u = None
        self.y = None
        self.sys = None
        self.K = None
        self.L = None
        self.T = None
        
        # Configurar estilo
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TCombobox', font=('Arial', 10))
        
        # Crear widgets
        self.create_widgets()
        
    def create_widgets(self):
        # Frame principal con paned window para permitir redimensionamiento
        main_paned = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=5)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Frame superior para controles y gráficos
        top_frame = ttk.Frame(main_paned)
        main_paned.add(top_frame)
        
        # Frame de controles
        control_frame = ttk.Frame(top_frame, padding="10", relief=tk.RAISED, borderwidth=2)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Botón para cargar archivo
        ttk.Label(control_frame, text="Archivo de datos:").grid(row=0, column=0, sticky=tk.W)
        self.file_entry = ttk.Entry(control_frame, width=50)
        self.file_entry.grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Examinar", command=self.load_file).grid(row=0, column=2)
        
        # Selector de modelo
        ttk.Label(control_frame, text="Modelo a identificar:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                         values=["Análisis respuesta escalón", "Lineal ARX", "No Lineal ARMAX", "No Lineal OE"])
        self.model_combobox.grid(row=1, column=1, sticky=tk.W, padx=5)
        self.model_combobox.current(0)
        ttk.Button(control_frame, text="Identificar", command=self.identify_model).grid(row=1, column=2)
        
        # Selector de método de diseño PID
        ttk.Label(control_frame, text="Método de diseño PID:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.method_var = tk.StringVar()
        self.method_combobox = ttk.Combobox(control_frame, textvariable=self.method_var, 
                                          values=["Ziegler-Nichols", "Lugar de las Raíces", "Respuesta en Frecuencia", "PID con Scikit-Learn", "PID Manual"])
        self.method_combobox.grid(row=2, column=1, sticky=tk.W, padx=5)
        self.method_combobox.current(0)
        ttk.Button(control_frame, text="Diseñar PID", command=self.design_pid).grid(row=2, column=2)
        
        # Frame para gráficos (ahora con paned window horizontal)
        graph_paned = tk.PanedWindow(top_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        graph_paned.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Frame para gráfico de modelo
        model_graph_frame = ttk.Frame(graph_paned)
        graph_paned.add(model_graph_frame)
        
        # Gráfico superior (modelo)
        self.fig_model = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax_model = self.fig_model.add_subplot(111)
        self.canvas_model = FigureCanvasTkAgg(self.fig_model, master=model_graph_frame)
        self.canvas_model.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Label(model_graph_frame, text="Identificación del Modelo", font=('Arial', 10, 'bold')).pack(side=tk.TOP)
        
        # Frame para gráfico de control
        control_graph_frame = ttk.Frame(graph_paned)
        graph_paned.add(control_graph_frame)
        
        # Gráfico inferior (control)
        self.fig_control = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax_control = self.fig_control.add_subplot(111)
        self.canvas_control = FigureCanvasTkAgg(self.fig_control, master=control_graph_frame)
        self.canvas_control.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Label(control_graph_frame, text="Control PID", font=('Arial', 10, 'bold')).pack(side=tk.TOP)
        
        # Frame inferior para logs (con mayor altura inicial)
        log_frame = ttk.Frame(main_paned, height=200)  # Altura inicial más grande
        main_paned.add(log_frame)
        
        # Configurar el área de logs
        log_header = ttk.Frame(log_frame)
        log_header.pack(fill=tk.X)
        ttk.Label(log_header, text="Registro de Eventos", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        
        # Botones para manejar logs
        btn_frame = ttk.Frame(log_header)
        btn_frame.pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="Limpiar", command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Guardar", command=self.save_logs).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Guardar PDF", command=self.save_pdf).pack(side=tk.LEFT)

        
        # Widget de texto para logs con scrollbar
        log_text_frame = ttk.Frame(log_frame)
        log_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(
            log_text_frame, 
            wrap=tk.WORD, 
            font=('Consolas', 9),  # Fuente monoespaciada para mejor alineación
            padx=5, 
            pady=5,
            height=10
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configurar colores para diferentes niveles de log
        self.log_text.tag_config('INFO', foreground='black')
        self.log_text.tag_config('WARNING', foreground='orange')
        self.log_text.tag_config('ERROR', foreground='red')
        self.log_text.tag_config('CRITICAL', foreground='red', underline=1)
        
        # Redirigir logs al widget de texto
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
            
            def emit(self, record):
                msg = self.format(record)
                tag = record.levelname
                
                self.text_widget.configure(state='normal')
                self.text_widget.insert(tk.END, msg + '\n', tag)
                self.text_widget.configure(state='disabled')
                self.text_widget.see(tk.END)
                self.text_widget.update()  # Actualizar la interfaz inmediatamente
        
        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(text_handler)
        
        # Configurar el peso de los paneles para que los gráficos tengan más espacio
        main_paned.paneconfig(top_frame, minsize=600)  # Mínimo 600px para la parte superior
        main_paned.paneconfig(log_frame, minsize=150)  # Mínimo 150px para los logs

    def clear_logs(self):
        """Limpiar el contenido del área de logs"""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
    
    def save_logs(self):
        """Guardar los logs en un archivo"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                logging.info(f"Logs guardados en: {filepath}")
            except Exception as e:
                logging.error(f"Error al guardar logs: {str(e)}")
    def save_pdf(self):
        """Guardar las gráficas y los logs en un archivo PDF"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        if filepath:
            try:
                with PdfPages(filepath) as pdf:
                    # Guardar la primera figura (modelo)
                    self.canvas_model.draw()
                    pdf.savefig(self.fig_model)

                    # Guardar la segunda figura (control)
                    self.canvas_control.draw()
                    pdf.savefig(self.fig_control)

                    # Crear figura para logs
                    fig_log = plt.Figure(figsize=(8.5, 11))
                    ax_log = fig_log.add_subplot(111)
                    logs = self.log_text.get(1.0, tk.END)
                    ax_log.axis('off')
                    ax_log.text(0, 1, logs, va='top', ha='left', wrap=True, fontsize=8)
                    pdf.savefig(fig_log)

                logging.info(f"Archivo PDF guardado en: {filepath}")
            except Exception as e:
                logging.error(f"Error al guardar el PDF: {str(e)}")
    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filepath:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filepath)
            try:
                self.t, self.u, self.y = self.cargar_datos(filepath)
                logging.info(f"Archivo cargado correctamente: {filepath}")
                self.plot_original_data()
            except Exception as e:
                logging.error(f"Error al cargar el archivo: {str(e)}")
    
    def cargar_datos(self, filename):
        try:
            data = np.loadtxt(filename, skiprows=1)
            t = data[:, 0]
            u = data[:, 1]
            y = data[:, 2]
            return t, u, y
        except Exception as e:
            logging.error(f"Error al leer el archivo: {str(e)}")
            raise
    
    def plot_original_data(self):
        if self.t is not None and self.u is not None and self.y is not None:
            self.ax_model.clear()
            self.ax_model.plot(self.t, self.y, label='Salida real')
            self.ax_model.plot(self.t, self.u, '--', label='Entrada')
            self.ax_model.set_xlabel('Tiempo (s)')
            self.ax_model.set_ylabel('Amplitud')
            self.ax_model.set_title('Datos Originales')
            self.ax_model.legend()
            self.ax_model.grid(True)
            self.canvas_model.draw()
            
    ##############################################
    # identificacion de modelos
    ##############################################
    def identify_model(self):
        if self.t is None or self.u is None or self.y is None:
            logging.error("No hay datos cargados para identificar el modelo")
            return
            
        model_type = self.model_var.get()
        try:
            if model_type == "Análisis respuesta escalón":
                self.sys, self.K, self.L, self.T = self.modelo_segundo_orden(self.t, self.u, self.y)
            elif model_type == "Lineal ARX":
                self.sys, self.K, self.L, self.T = self.modelo_lineal(self.t, self.u, self.y)
            elif model_type == "No Lineal ARMAX":
                self.sys, self.K, self.L, self.T = self.modelo_nolineal(self.t, self.u, self.y)
            elif model_type == "No Lineal OE":
                self.sys, self.K, self.L, self.T = self.modelo_nolineal_oe(self.t, self.u, self.y)
            else:
                logging.error("Tipo de modelo no reconocido")
                return
                
            logging.info(f"Modelo {model_type} identificado correctamente")
        except Exception as e:
            logging.error(f"Error al identificar el modelo: {str(e)}")
    
    def modelo_segundo_orden(self, t, u, y):
        K = y[-1] / u[-1]
        target_63 = 0.632 * K * u[-1]
        target_28 = 0.283 * K * u[-1]
        tau_63 = t[np.argmax(y >= target_63)]
        tau_28 = t[np.argmax(y >= target_28)]

        T = 1.5 * (tau_63 - tau_28)
        zeta = (1.733 * (tau_63 - tau_28)) / T if T != 0 else 1

        if zeta < 1:
            wn = 1 / (tau_63 * np.sqrt(1 - zeta**2)) if tau_63 != 0 else 1
            num = [K * wn**2]
            den = [1, 2*zeta*wn, wn**2]
        elif zeta == 1:
            a = 1 / tau_63 if tau_63 != 0 else 1
            num = [K * a**2]
            den = [1, 2*a, a**2]
        else:
            a1 = 1 / tau_63 if tau_63 != 0 else 1
            a2 = 1 / tau_28 if tau_28 != 0 else 1
            num = [K * a1 * a2]
            den = [1, a1 + a2, a1 * a2]

        sys = ct.TransferFunction(num, den)
        t_model, y_model = ct.step_response(sys, T=t[-1], T_num=len(t))
        y_model *= u[-1]

        self.plot_model_comparison(t, y, t_model, y_model, 'Modelo estimado (análisis respuesta escalón)')
        logging.info("\nFunción de transferencia estimada:\n%s", sys)

        mse = self.calcular_mse(y, y_model)
        logging.info("Error cuadrático medio (2º orden): %.6f", mse)
        return sys, K, tau_28, tau_63 - tau_28
    
    def modelo_nolineal_oe(self, t, u, y, nb=2, d=1):
        """
        Modelo OE no lineal (cuadrático) con tiempo muerto de d pasos.
        :param t: Tiempo
        :param u: Entrada
        :param y: Salida
        :param nb: Orden del numerador (entrada)
        :param d: Tiempo muerto (en pasos)
        """
        N = len(y)
        inicio = nb + d
        Phi = []

        for i in range(inicio, N):
            row = []
            # Términos lineales en u con tiempo muerto
            for j in range(1, nb + 1):
                row.append(u[i - d - j + 1])
            # Términos no lineales en u^2
            for j in range(1, nb + 1):
                row.append(u[i - d - j + 1] ** 2)
            Phi.append(row)

        Phi = np.array(Phi)
        Y = y[inicio:]

        # Estimación de parámetros
        theta = np.linalg.lstsq(Phi, Y, rcond=None)[0]
        logging.info("\nCoeficientes estimados OE no lineal:\n%s", theta)

        # Simulación del modelo OE
        y_sim = np.zeros(N)
        y_sim[:inicio] = y[:inicio]
        for i in range(inicio, N):
            val = 0
            idx = 0
            for j in range(nb):
                val += theta[idx] * u[i - d - j]
                idx += 1
            for j in range(nb):
                val += theta[idx] * u[i - d - j] ** 2
                idx += 1
            y_sim[i] = val

        self.plot_model_comparison(t, y, t, y_sim, f'Modelo OE no lineal (d = {d} pasos)')
        mse = self.calcular_mse(y, y_sim, offset=inicio)
        logging.info("Error cuadrático medio (OE no lineal): %.6f", mse)

        # Aproximación para FT (sólo para representar ganancia y retardo)
        Ts = t[1] - t[0]
        K = y[-1] / u[-1] if u[-1] != 0 else 1
        num = [0]*d + [K]
        den = [1, 1]  # Aproximación básica
        sys = ct.TransferFunction(num, den, dt=Ts)

        return sys, K, d * Ts, Ts * 2
    
    def modelo_lineal(self, t, u, y, d=1):
        """
        Modelo ARX de segundo orden con tiempo muerto (d pasos).
        :param t: Vector de tiempo
        :param u: Entrada (input)
        :param y: Salida (output)
        :param d: Tiempo muerto en pasos (entero)
        """
        N = len(y)
        if N <= d + 2:
            raise ValueError("La señal es muy corta para el tiempo muerto especificado.")

        # Matriz de regresores φ y vector de salida Y
        phi = np.zeros((N - (d + 2), 4))
        for i in range(d + 2, N):
            phi[i - (d + 2)] = [y[i - 1], y[i - 2], u[i - d - 1], u[i - d - 2]]
        Y = y[d + 2:]

        # Estimación de parámetros (a1, a2, b1, b2)
        theta = np.linalg.lstsq(phi, Y, rcond=None)[0]
        logging.info("\nParámetros estimados (a1, a2, b1, b2):\n%s", theta)

        # Simulación del modelo ARX con tiempo muerto
        y_sim = np.zeros(N)
        y_sim[:d + 2] = y[:d + 2]  # condiciones iniciales
        for i in range(d + 2, N):
            y_sim[i] = (
                theta[0] * y_sim[i - 1] +
                theta[1] * y_sim[i - 2] +
                theta[2] * u[i - d - 1] +
                theta[3] * u[i - d - 2]
            )

        self.plot_model_comparison(t, y, t, y_sim, f'Modelo ARX (d = {d} pasos)')
        mse = self.calcular_mse(y, y_sim, offset=d + 2)
        logging.info("Error cuadrático medio (ARX): %.6f", mse)

        # Crear función de transferencia discreta con tiempo muerto
        num = [0]*d + [theta[2], theta[3]]  # incluir ceros para representar el delay
        den = [1, -theta[0], -theta[1]]
        Ts = t[1] - t[0]
        sys = ct.TransferFunction(num, den, dt=Ts)

        return sys, sum(num)/sum(den), Ts, d*Ts
    
    def modelo_nolineal(self, t, u, y, na=2, nb=2):
        N = len(y)
        rows = N - max(na, nb)
        Phi = []

        for i in range(max(na, nb), N):
            row = []
            for j in range(1, na+1): row.append(y[i-j])
            for j in range(1, nb+1): row.append(u[i-j])
            for j in range(1, na+1): row.append(y[i-j]**2)
            for j in range(1, nb+1): row.append(u[i-j]**2)
            Phi.append(row)

        Phi = np.array(Phi)
        Y = y[max(na, nb):]
        theta = np.linalg.lstsq(Phi, Y, rcond=None)[0]
        logging.info("\nCoeficientes estimados (no lineal):\n%s", theta)

        y_sim = np.zeros(N)
        y_sim[:max(na, nb)] = y[:max(na, nb)]

        for i in range(max(na, nb), N):
            val = 0
            idx = 0
            for j in range(na):
                val += theta[idx] * y_sim[i-j-1]
                idx += 1
            for j in range(nb):
                val += theta[idx] * u[i-j-1]
                idx += 1
            for j in range(na):
                val += theta[idx] * y_sim[i-j-1]**2
                idx += 1
            for j in range(nb):
                val += theta[idx] * u[i-j-1]**2
                idx += 1
            y_sim[i] = val

        self.plot_model_comparison(t, y, t, y_sim, 'Modelo no lineal (polinomial)')
        mse = self.calcular_mse(y, y_sim, offset=max(na, nb))
        logging.info("Error cuadrático medio (no lineal): %.6f", mse)
        K = y[-1] / u[-1]
        L = t[np.argmax(y > 0.02 * y[-1])]
        T_p = t[np.argmax(y > 0.7 * y[-1])] - L
        num = [K]
        den = [T_p, 1]
        sys = ct.TransferFunction(num, den)
        return sys, K, L, T_p
    
    def manual_pid_design(self):
        """Muestra una ventana para ingresar manualmente los parámetros PID"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configuración Manual PID")
        dialog.geometry("300x200")
        dialog.resizable(False, False)
        
        # Centrar la ventana
        window_width = 300
        window_height = 200
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)
        dialog.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
        
        # Variables para los valores PID
        Kp_var = tk.DoubleVar(value=1.0)
        Ki_var = tk.DoubleVar(value=0.1)
        Kd_var = tk.DoubleVar(value=0.01)
        
        # Etiquetas y campos de entrada
        ttk.Label(dialog, text="Ganancia Proporcional (Kp):").pack(pady=(10, 0))
        Kp_entry = ttk.Entry(dialog, textvariable=Kp_var)
        Kp_entry.pack()
        
        ttk.Label(dialog, text="Ganancia Integral (Ki):").pack(pady=(5, 0))
        Ki_entry = ttk.Entry(dialog, textvariable=Ki_var)
        Ki_entry.pack()
        
        ttk.Label(dialog, text="Ganancia Derivativa (Kd):").pack(pady=(5, 0))
        Kd_entry = ttk.Entry(dialog, textvariable=Kd_var)
        Kd_entry.pack()
        
        # Botón para aplicar
        def apply_pid():
            Kp = Kp_var.get()
            Ki = Ki_var.get()
            Kd = Kd_var.get()
            
            if Kp < 0 or Ki < 0 or Kd < 0:
                tk.messagebox.showerror("Error", "Los valores PID no pueden ser negativos")
                return
                
            self.simular_controlador_pid(Kp, Ki, Kd, "PID Manual")
            dialog.destroy()
        
        ttk.Button(dialog, text="Aplicar", command=apply_pid).pack(pady=10)
        
        # Hacer que la ventana sea modal
        dialog.grab_set()
        self.root.wait_window(dialog)

    ##############################################
    # Métodos para diseñar el controlador PID
    ##############################################
    def design_pid(self):
        if self.sys is None:
            logging.error("No hay modelo identificado para diseñar el controlador")
            return
            
        method = self.method_var.get()
        try:
            if method == "Ziegler-Nichols":
                Kp, Ki, Kd = self.ziegler_nichols_pid(self.K, self.L, self.T)
                self.simular_controlador_pid(Kp, Ki, Kd, "Ziegler-Nichols")
            elif method == "Lugar de las Raíces":
                Kp, Ki, Kd = self.disenar_pid_lugar_raices(self.sys)
                self.simular_controlador_pid(Kp, Ki, Kd, "Lugar de Raíces")
                self.graficar_lugar_raices(self.sys, Kp, Ki, Kd)
            elif method == "Respuesta en Frecuencia":
                Kp, Ki, Kd = self.disenar_pid_bode(self.sys)
                self.simular_controlador_pid(Kp, Ki, Kd, "Respuesta en Frecuencia")
                self.graficar_bode(self.sys, Kp, Ki, Kd)
            elif method == "PID con Scikit-Learn":
                if self.t is None or self.y is None or self.u is None:
                    logging.error("Datos insuficientes para entrenar PID con scikit-learn")
                    return
                dt = self.t[1] - self.t[0]
                setpoint = np.ones_like(self.y) * self.y[-1]  # Asumimos setpoint constante
                Kp, Ki, Kd = self.pid_via_sklearn(self.sys)
                self.simular_controlador_pid(Kp, Ki, Kd, "PID con Scikit-Learn")
            elif method == "PID Manual":
                self.manual_pid_design()
            else:
                logging.error("Método de diseño no reconocido")
        except Exception as e:
            logging.error(f"Error al diseñar el controlador PID: {str(e)}")

    def plot_model_comparison(self, t, y_real, t_model, y_model, titulo):
        self.ax_model.clear()
        self.ax_model.plot(t, y_real, label='Salida real')
        self.ax_model.plot(t_model, y_model, '--', label='Salida modelo')
        self.ax_model.set_xlabel('Tiempo (s)')
        self.ax_model.set_ylabel('Amplitud')
        self.ax_model.set_title(titulo)
        self.ax_model.legend()
        self.ax_model.grid(True)
        self.canvas_model.draw()

    def calcular_mse(self, y_real, y_modelo, offset=0):
        return np.mean((y_real[offset:] - y_modelo[offset:])**2)
    
    def ziegler_nichols_pid(self, K, L, T):
        Kp = 1.2 * T / (K * L)
        Ti = 2 * L
        Td = 0.5 * L
        Ki = Kp / Ti
        Kd = Kp * Td
        return Kp, Ki, Kd

    def disenar_pid_lugar_raices(self, planta, factor_amortiguamiento=0.7, margen_fase=60):
        if planta.dt is not None and planta.dt > 0:
            planta_continua = planta
        else:
            planta_continua = planta
        
        polos = ct.poles(planta_continua)
        ceros = ct.zeros(planta_continua)
        
        logging.info("Polos de la planta: %s", polos)
        logging.info("Ceros de la planta: %s", ceros)
        
        ts_deseado = 2.0
        wn_deseado = 4 / (factor_amortiguamiento * ts_deseado)
        
        if planta.dt is not None and planta.dt > 0:
            T = planta.dt
            wn_deseado = min(wn_deseado, np.pi/(4*T))
        
        if planta.dt is not None and planta.dt > 0:
            s_continuo = -factor_amortiguamiento * wn_deseado + 1j * wn_deseado * np.sqrt(1 - factor_amortiguamiento**2)
            polo_deseado = np.exp(s_continuo * planta.dt)
        else:
            polo_deseado = -factor_amortiguamiento * wn_deseado + 1j * wn_deseado * np.sqrt(1 - factor_amortiguamiento**2)
        
        logging.info("Polo dominante deseado: %s", polo_deseado)
        logging.info("Frecuencia natural deseada: %.3f rad/s", wn_deseado)
        logging.info("Factor de amortiguamiento deseado: %s", factor_amortiguamiento)
        
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
            
            Kp = wn_deseado**2 / (K_planta * abs(polos[0])**2)
            Ki = wn_deseado**3 / (K_planta * abs(polos[0])**2)
            Kd = 2 * factor_amortiguamiento * wn_deseado / (K_planta * abs(polos[0]))
        
        Kp = abs(Kp)
        Ki = abs(Ki)
        Kd = abs(Kd)
        
        return Kp, Ki, Kd
    
    def disenar_pid_bode(self, planta, margen_fase_deseado=60, wc_deseado=None):
        # Mostrar diagrama de Bode de la planta
        mag, phase, omega = ct.bode_plot(planta, dB=True, plot=True)
        plt.suptitle("Diagrama de Bode de la planta")  # Título general
        plt.show()

        # Crear variable simbólica 's' según tipo de sistema
        if planta.dt is not None and planta.dt > 0:
            s = ct.TransferFunction([1, 0], [1], dt=planta.dt)
        else:
            s = ct.TransferFunction([1, 0], [1])

        # Obtener ganancia estática
        K_planta = abs(ct.dcgain(planta))

        # Obtener respuesta en frecuencia para encontrar frecuencia de cruce
        omega_eval = np.logspace(-2, 2, 1000)  # Frecuencia de 0.01 a 100 rad/s
        response = ct.frequency_response(planta, omega_eval)  # Respuesta compleja
        mag = np.abs(response)

        if wc_deseado is None:
            indices = np.where(mag >= 1.0)[0]
            if len(indices) > 0:
                wc_deseado = omega_eval[indices[0]]
            else:
                wc_deseado = omega_eval[-1]  # Usa la máxima si no cruza

        # Diseño PID básico (puedes mejorarlo con técnicas específicas)
        Kp = 1.0 / K_planta
        Ki = 0.1 * Kp * wc_deseado
        Kd = 0.1 * Kp / wc_deseado

        return Kp, Ki, Kd
    
    def pid_via_sklearn(self, planta, n_estimators=100, test_size=0.2, random_state=42):
        """
        Diseña un controlador PID utilizando Random Forest de scikit-learn
        para predecir los parámetros óptimos del PID basado en características
        de la planta.
        
        Args:
            planta: Objeto de la planta (control.TransferFunction)
            n_estimators: Número de árboles en el Random Forest
            test_size: Proporción de datos para prueba
            random_state: Semilla para reproducibilidad
            
        Returns:
            Kp, Ki, Kd: Parámetros del controlador PID
        """
        # 1. Extraer características de la planta
        polos = ct.poles(planta)
        ceros = ct.zeros(planta)
        K_planta = abs(ct.dcgain(planta))
        
        # Características para el modelo
        features = []
        targets = []
        
        # 2. Generar datos sintéticos basados en diseño clásico de PID
        # (En una implementación real, usarías datos históricos o de simulación)
        for _ in range(1000):
            # Variar parámetros de diseño
            factor_amort = np.random.uniform(0.5, 0.9)
            ts_deseado = np.random.uniform(1.0, 5.0)
            
            # Calcular parámetros PID con método clásico (como referencia)
            if len(polos) == 1:
                Kp = 0.5 / K_planta
                Ki = Kp / 10
                Kd = Kp / 10
            else:
                wn_deseado = 4 / (factor_amort * ts_deseado)
                Kp = wn_deseado**2 / (K_planta * abs(polos[0])**2)
                Ki = wn_deseado**3 / (K_planta * abs(polos[0])**2)
                Kd = 2 * factor_amort * wn_deseado / (K_planta * abs(polos[0]))
            
            # Características de la planta para este caso
            plant_features = [
                len(polos),
                len(ceros),
                K_planta,
                np.mean([abs(p) for p in polos]),
                np.std([abs(p) for p in polos]),
                factor_amort,
                ts_deseado
            ]
            
            features.append(plant_features)
            targets.append([Kp, Ki, Kd])
        
        features = np.array(features)
        targets = np.array(targets)
        
        # 3. Entrenar modelo de Random Forest
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=random_state
        )
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        
        # 4. Predecir parámetros PID para nuestra planta actual
        current_plant_features = np.array([
            len(polos),
            len(ceros),
            K_planta,
            np.mean([abs(p) for p in polos]),
            np.std([abs(p) for p in polos]),
            0.7,  # factor_amortiguamiento típico
            2.0   # ts_deseado típico
        ]).reshape(1, -1)
        
        Kp, Ki, Kd = model.predict(current_plant_features)[0]
        
        # 5. Evaluar modelo (opcional, para depuración)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"Error cuadrático medio del modelo: {mse:.4f}")
        
        # Asegurar parámetros positivos
        Kp = abs(Kp)
        Ki = abs(Ki)
        Kd = abs(Kd)
        
        logging.info(f"Parámetros PID predichos: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}")
        
        return Kp, Ki, Kd

    def simular_controlador_pid(self, Kp, Ki, Kd, method_name):
        if self.sys is None or self.t is None or self.u is None:
            logging.error("Datos insuficientes para simular el controlador")
            return
            
        pid = self.crear_controlador_pid(Kp, Ki, Kd, self.sys.dt if self.sys.dt is not None else None)
        
        if self.sys.dt is not None and self.sys.dt > 0:
            dt = self.sys.dt
            T_sim = np.arange(0, self.t[-1] + dt, dt)
        else:
            T_sim = self.t

        sistema_abierto = ct.series(pid, self.sys)
        sistema_cerrado = ct.feedback(sistema_abierto, 1)

        t_out, y_out = ct.step_response(sistema_cerrado, T=T_sim)
        y_out *= 3.5
        
        y_interp = np.interp(t_out, self.t, self.y)
        
        self.ax_control.clear()
        self.ax_control.plot(t_out, y_interp, label='Salida real')
        self.ax_control.plot(t_out, y_out, '--', label=f'Control PID ({method_name})')
        self.ax_control.set_xlabel('Tiempo (s)')
        self.ax_control.set_ylabel('Amplitud')
        self.ax_control.set_title(f'Control PID - {method_name}')
        self.ax_control.legend()
        self.ax_control.grid(True)
        self.canvas_control.draw()
        
        logging.info(f"\nControlador PID ({method_name}):")
        logging.info("Kp = %.6f", Kp)
        logging.info("Ki = %.6f", Ki)
        logging.info("Kd = %.6f", Kd)
        logging.info("Función de transferencia del controlador:\n%s", pid)
        
        self.analizar_estabilidad_margenes(self.sys, pid, method_name)
    
    def crear_controlador_pid(self, Kp, Ki, Kd, dt=None):
        if dt is not None and dt > 0:
            T = dt
            c2 = Kp*T**2 + Ki*T**2 + 2*Kd*T
            c1 = 2*Ki*T**2 - 4*Kd*T
            c0 = Kp*T**2 + Ki*T**2 + 2*Kd*T
            
            num = [c2, c1, c0]
            den = [T**2, 0, -T**2]
            
            pid = ct.TransferFunction(num, den, dt=dt)
        else:
            num = [Kd, Kp, Ki]
            den = [1, 0]
            pid = ct.TransferFunction(num, den)
        
        return pid
    
    def graficar_lugar_raices(self, planta, Kp, Ki, Kd):
        pid = self.crear_controlador_pid(Kp, Ki, Kd, planta.dt if planta.dt is not None else None)
        
        try:
            sistema_abierto = ct.series(pid, planta)
            
            plt.figure(figsize=(8, 6))
            ct.rlocus(sistema_abierto, plot=True)
            plt.title('Lugar de las Raíces')
            plt.grid(True)
            plt.xlabel('Parte Real')
            plt.ylabel('Parte Imaginaria')
            
            sistema_cerrado = ct.feedback(sistema_abierto, 1)
            polos_cerrados = ct.poles(sistema_cerrado)
            
            for polo in polos_cerrados:
                plt.plot(polo.real, polo.imag, 'rx', markersize=10, markeredgewidth=2)
            
            plt.legend(['Lugar de raíces', 'Polos lazo cerrado'])
            plt.show()
            
            logging.info("Polos de lazo cerrado:")
            for i, polo in enumerate(polos_cerrados):
                logging.info("  Polo %d: %s", i+1, polo)
                
        except Exception as e:
            logging.error("Error al graficar lugar de raíces: %s", e)

    def graficar_bode(self, planta, Kp, Ki, Kd):
        pid = self.crear_controlador_pid(Kp, Ki, Kd, planta.dt if planta.dt is not None else None)
        
        try:
            sistema_abierto = ct.series(pid, planta)
            
            plt.figure(figsize=(10, 8))
            ct.bode_plot(sistema_abierto, dB=True, margins=True)
            plt.suptitle('Diagrama de Bode')
            plt.show()
            
            gm, pm, wg, wp = ct.margin(sistema_abierto)
            logging.info("\n--- Análisis de Estabilidad ---")
            logging.info("Margen de Ganancia: %.3f (%.2f dB)", gm, 20*np.log10(gm))
            logging.info("Margen de Fase: %.2f°", pm)
            logging.info("Frecuencia de cruce de ganancia: %.3f rad/s", wg)
            logging.info("Frecuencia de cruce de fase: %.3f rad/s", wp)
            
        except Exception as e:
            logging.error("Error al graficar diagrama de Bode: %s", e)

    def analizar_estabilidad_margenes(self, planta, controlador_pid, titulo):
        try:
            sistema_abierto = ct.series(controlador_pid, planta)
            
            gm, pm, wg, wp = ct.margin(sistema_abierto)
            
            logging.info("\n--- Análisis de Estabilidad - %s ---", titulo)
            logging.info("Margen de Ganancia: %.3f (%.2f dB)", gm, 20*np.log10(gm))
            logging.info("Margen de Fase: %.2f°", pm)
            logging.info("Frecuencia de cruce de ganancia: %.3f rad/s", wg)
            logging.info("Frecuencia de cruce de fase: %.3f rad/s", wp)
            
            sistema_cerrado = ct.feedback(sistema_abierto, 1)
            polos_cerrados = ct.poles(sistema_cerrado)
            
            if planta.dt is not None and planta.dt > 0:
                estable = all(abs(polo) < 1 for polo in polos_cerrados)
            else:
                estable = all(polo.real < 0 for polo in polos_cerrados)
            
            logging.info("Sistema estable: %s", 'Sí' if estable else 'No')
            
            return gm, pm, wg, wp, estable
            
        except Exception as e:
            logging.error("Error en análisis de estabilidad para %s: %s", titulo, e)
            return None, None, None, None, False

if __name__ == "__main__":
    root = tk.Tk()
    app = PIDControlApp(root)
    root.mainloop()