"""
PROYECTO-NATHY_IA - Interfaz grafica con INSTRUCTOR
-----------------------------------------------------
Una ventana (tkinter, viene con Python) con dos modos de aprendizaje y un panel de
instructor a la derecha que explica, en lenguaje simple, lo que esta pasando:

  1) Supervisado (LSTM): genera letras y permite COMPARAR el modelo anterior (que
     memorizaba) con el nuevo. Muestra cuanto copia cada uno y, paso a paso, que
     palabras barajo el modelo antes de elegir.
  2) No supervisado (agrupamiento): K-Means agrupa las canciones por su cuenta y
     grafica los grupos, el metodo del codo y las metricas.

Se corre desde la raiz del proyecto:
    python src/interfaz.py
"""

import os
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox, scrolledtext

from generar_texto import (generar_con_detalle, modelo_disponible, MODELOS,
                           PALABRAS_A_GENERAR, TEMPERATURA)
from diagnostico import diagnosticar

CARPETA_SALIDA_TEXTO = "outputs/textos_generados"
RUTA_ULTIMA_GENERACION = os.path.join(CARPETA_SALIDA_TEXTO, "ultima_generacion.txt")

# --- Colores ---
FONDO = "#f6f4ef"
PANEL = "#ffffff"
TINTA = "#23262b"
SUAVE = "#6b7078"
LINEA = "#d9d5cb"
ACENTO = "#2f6f6a"        # verde azulado
ACENTO_OSC = "#235651"
INSTRUCTOR_FONDO = "#fbf6e8"
AVISO = "#fdeadf"
COLOR_ANTERIOR = "#b4532a"
COLOR_NUEVO = "#2f6f6a"
COLORES_GRUPO = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


def ver(palabra: str) -> str:
    """Como se muestra un token en pantalla."""
    return {"|": "¶", "<PAD>": "·", "<UNK>": "?"}.get(palabra, palabra)


class ZonaDesplazable(tk.Frame):
    """Marco con barra de desplazamiento vertical (y rueda del raton) para pantallas chicas.
    Si el contenido cabe, se estira para llenar el espacio; si no cabe, aparece la barra."""

    def __init__(self, padre, fondo):
        super().__init__(padre, bg=fondo)
        self.canvas = tk.Canvas(self, bg=fondo, highlightthickness=0, bd=0)
        self.barra = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.barra.set)
        self.barra.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.interior = tk.Frame(self.canvas, bg=fondo)
        self._ventana = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")
        self.interior.bind("<Configure>", self._ajustar)
        self.canvas.bind("<Configure>", self._ajustar)
        for evento in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind_all(evento, self._rueda, add="+")
        self.after(300, self._vigilar)

    def _ajustar(self, _evento=None):
        ancho, alto = self.canvas.winfo_width(), self.canvas.winfo_height()
        necesario = self.interior.winfo_reqheight()
        # si sobra espacio, el contenido se estira; si falta, toma su altura natural y se desplaza
        self.canvas.itemconfigure(self._ventana, width=ancho, height=alto if alto > necesario else 0)
        self.canvas.configure(scrollregion=(0, 0, ancho, max(alto, necesario)))

    def _vigilar(self):
        """Recalcula la altura cada rato: el contenido cambia de tamano al generar letras."""
        self._ajustar()
        self.after(300, self._vigilar)

    def _dentro(self, widget):
        while widget is not None:
            if widget is self:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _rueda(self, evento):
        bajo_el_puntero = self.winfo_containing(evento.x_root, evento.y_root)
        if bajo_el_puntero is None or not self._dentro(bajo_el_puntero):
            return
        if bajo_el_puntero.winfo_class() in ("Text", "Treeview", "Listbox"):
            return   # esos widgets ya se desplazan solos
        if self.interior.winfo_reqheight() <= self.canvas.winfo_height():
            return
        sentido = -1 if (evento.num == 4 or getattr(evento, "delta", 0) > 0) else 1
        self.canvas.yview_scroll(sentido * 3, "units")

    def al_inicio(self):
        self.canvas.yview_moveto(0)


class App:
    def __init__(self, ventana):
        self.ventana = ventana
        ventana.title("PROYECTO-NATHY_IA · Generador de letras con instructor")
        ancho = min(1280, ventana.winfo_screenwidth() - 30)
        alto = min(820, ventana.winfo_screenheight() - 90)
        ventana.geometry(f"{ancho}x{alto}+10+10")   # nunca mas grande que tu pantalla
        ventana.minsize(700, 480)
        ventana.configure(bg=FONDO)

        familias = set(tkfont.families())
        self.f = "Segoe UI" if "Segoe UI" in familias else ("Helvetica" if "Helvetica" in familias else "TkDefaultFont")
        self._estilos()

        self.modo = tk.StringVar(value="supervisado")
        self.eleccion = tk.StringVar(value="comparar")
        self.modelo_pasos = tk.StringVar(value="nuevo")
        self.resultados = {}        # {"nuevo": {...}, "anterior": {...}}
        self.ocupado = False
        self.fig = None
        self.resultado_grupos = None
        self.cola = queue.Queue()   # los hilos dejan aqui sus resultados; la ventana los recoge

        self._construir_encabezado()
        self._construir_cuerpo()
        self._construir_pie()
        self.cambiar_modo()
        self._revisar_cola()

    # ------------------------------------------------------------------ estilos
    def _estilos(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=FONDO, foreground=TINTA, font=(self.f, 10))
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabelframe", background=PANEL, bordercolor=LINEA)
        s.configure("TLabelframe.Label", background=PANEL, foreground=SUAVE, font=(self.f, 9, "bold"))
        s.configure("Panel.TLabel", background=PANEL)
        s.configure("Suave.TLabel", background=PANEL, foreground=SUAVE, font=(self.f, 9))
        s.configure("Accion.TButton", background=ACENTO, foreground="white", font=(self.f, 10, "bold"),
                    padding=(14, 7), borderwidth=0)
        s.map("Accion.TButton", background=[("active", ACENTO_OSC), ("disabled", "#a9b9b6")])
        s.configure("Sec.TButton", background="#e8e4d9", foreground=TINTA, font=(self.f, 10), padding=(12, 7), borderwidth=0)
        s.map("Sec.TButton", background=[("active", "#dcd7c8"), ("disabled", "#efece4")])
        s.configure("Modo.TRadiobutton", background=FONDO, font=(self.f, 11, "bold"), padding=(10, 6))
        s.configure("Panel.TRadiobutton", background=PANEL)
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL, rowheight=24, bordercolor=LINEA)
        s.configure("Treeview.Heading", font=(self.f, 9, "bold"), background="#ece8dd")
        s.map("Treeview", background=[("selected", "#cfe3e0")], foreground=[("selected", TINTA)])
        s.configure("Horizontal.TScale", background=PANEL)

    # ------------------------------------------------------------------ estructura
    def _construir_encabezado(self):
        enc = tk.Frame(self.ventana, bg=FONDO)
        enc.pack(fill="x", padx=18, pady=(14, 6))
        tk.Label(enc, text="Generador de letras", bg=FONDO, fg=TINTA, font=(self.f, 18, "bold")).pack(side="left")
        tk.Label(enc, text="  con instructor", bg=FONDO, fg=SUAVE, font=(self.f, 18)).pack(side="left")
        modos = tk.Frame(self.ventana, bg=FONDO)
        modos.pack(fill="x", padx=18, pady=(0, 2))
        tk.Label(modos, text="Modo de aprendizaje:", bg=FONDO, fg=SUAVE, font=(self.f, 10)).pack(side="left", padx=(0, 6))
        ttk.Radiobutton(modos, text="Supervisado · LSTM", variable=self.modo, value="supervisado",
                        style="Modo.TRadiobutton", command=self.cambiar_modo).pack(side="left")
        ttk.Radiobutton(modos, text="No supervisado · Agrupamiento", variable=self.modo, value="no_supervisado",
                        style="Modo.TRadiobutton", command=self.cambiar_modo).pack(side="left")

    def _construir_cuerpo(self):
        cuerpo = tk.PanedWindow(self.ventana, orient="horizontal", bg=FONDO, sashwidth=6, bd=0, sashrelief="flat")
        cuerpo.pack(fill="both", expand=True, padx=18, pady=(4, 6))

        self.desplazable = ZonaDesplazable(cuerpo, FONDO)
        self.zona = self.desplazable.interior
        cuerpo.add(self.desplazable, stretch="always", minsize=480)

        caja = tk.Frame(cuerpo, bg=INSTRUCTOR_FONDO, highlightbackground=LINEA, highlightthickness=1)
        ancho_instructor = 290 if self.ventana.winfo_screenwidth() < 1300 else 380   # en pantallas chicas, mas espacio para el trabajo
        cuerpo.add(caja, minsize=260, width=ancho_instructor)
        cab = tk.Frame(caja, bg=INSTRUCTOR_FONDO)
        cab.pack(fill="x", padx=14, pady=(12, 0))
        tk.Label(cab, text="Instructor", bg=INSTRUCTOR_FONDO, fg=TINTA, font=(self.f, 13, "bold")).pack(side="left")
        marco_texto = tk.Frame(caja, bg=INSTRUCTOR_FONDO)
        marco_texto.pack(fill="both", expand=True)
        barra_instructor = ttk.Scrollbar(marco_texto, orient="vertical")
        barra_instructor.pack(side="right", fill="y")
        self.texto_instructor = tk.Text(marco_texto, wrap="word", bg=INSTRUCTOR_FONDO, fg=TINTA, relief="flat",
                                        font=(self.f, 10), padx=14, pady=8, spacing1=2, spacing3=4, cursor="arrow",
                                        highlightthickness=0, yscrollcommand=barra_instructor.set)
        self.texto_instructor.pack(side="left", fill="both", expand=True)
        barra_instructor.config(command=self.texto_instructor.yview)
        t = self.texto_instructor
        t.tag_configure("h", font=(self.f, 11, "bold"), spacing1=10, spacing3=2, foreground=ACENTO_OSC)
        t.tag_configure("p", font=(self.f, 10), spacing3=4)
        t.tag_configure("tip", background="#e4efe7", lmargin1=8, lmargin2=8, rmargin=8, spacing1=6, spacing3=6)
        t.tag_configure("aviso", background=AVISO, lmargin1=8, lmargin2=8, rmargin=8, spacing1=6, spacing3=6)
        t.tag_configure("dato", font=(self.f, 10, "bold"))
        t.config(state="disabled")

        self._construir_panel_supervisado()
        self._construir_panel_no_supervisado()

    def _construir_pie(self):
        self.estado = tk.Label(self.ventana, text="Listo.", bg=FONDO, fg=SUAVE, anchor="w", font=(self.f, 9))
        self.estado.pack(fill="x", padx=20, pady=(0, 8))

    # ------------------------------------------------------------------ panel supervisado
    def _construir_panel_supervisado(self):
        p = self.panel_sup = tk.Frame(self.zona, bg=FONDO)

        # Controles
        ctrl = ttk.Frame(p, style="Panel.TFrame", padding=12)
        ctrl.pack(fill="x")
        ttk.Label(ctrl, text="Línea semilla (con qué empieza la letra)", style="Suave.TLabel").grid(row=0, column=0, sticky="w", columnspan=4)
        self.entrada_semilla = ttk.Entry(ctrl, font=(self.f, 11))
        self.entrada_semilla.grid(row=1, column=0, columnspan=4, sticky="we", pady=(2, 8))
        self.entrada_semilla.insert(0, "hoy te extraño tanto")
        self.entrada_semilla.bind("<Return>", lambda e: self.generar())

        self.temperatura = tk.DoubleVar(value=TEMPERATURA)
        self.palabras = tk.IntVar(value=PALABRAS_A_GENERAR)
        self._control_deslizante(ctrl, 2, 0, "Temperatura (atrevimiento)", self.temperatura, 0.3, 1.5, "{:.2f}")
        self._control_deslizante(ctrl, 2, 2, "Palabras a generar", self.palabras, 10, 80, "{:.0f}")

        opciones = ttk.Frame(ctrl, style="Panel.TFrame")
        opciones.grid(row=4, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Label(opciones, text="Modelo:", style="Panel.TLabel").pack(side="left", padx=(0, 8))
        for valor, texto in (("comparar", "Comparar los dos"), ("nuevo", "Solo el nuevo"), ("anterior", "Solo el anterior")):
            ttk.Radiobutton(opciones, text=texto, value=valor, variable=self.eleccion,
                            style="Panel.TRadiobutton", command=self._ajustar_columnas).pack(side="left", padx=6)

        botones = ttk.Frame(ctrl, style="Panel.TFrame")
        botones.grid(row=5, column=0, columnspan=4, sticky="w", pady=(12, 0))
        self.boton_generar = ttk.Button(botones, text="Generar letra", style="Accion.TButton", command=self.generar)
        self.boton_generar.pack(side="left")
        self.boton_escuchar = ttk.Button(botones, text="Escuchar la del modelo nuevo", style="Sec.TButton",
                                         command=self.escuchar, state="disabled")
        self.boton_escuchar.pack(side="left", padx=10)
        ctrl.columnconfigure(0, weight=1)
        ctrl.columnconfigure(2, weight=1)

        # Resultados (dos columnas)
        self.cols = tk.Frame(p, bg=FONDO)
        self.cols.pack(fill="both", expand=True, pady=(10, 0))
        self.cols.columnconfigure(0, weight=1, uniform="c")
        self.cols.columnconfigure(1, weight=1, uniform="c")
        self.cols.rowconfigure(0, weight=1)
        self.col = {}
        for i, (clave, color) in enumerate((("anterior", COLOR_ANTERIOR), ("nuevo", COLOR_NUEVO))):
            marco = tk.Frame(self.cols, bg=PANEL, highlightbackground=LINEA, highlightthickness=1)
            marco.grid(row=0, column=i, sticky="nsew", padx=(0, 6) if i == 0 else (6, 0))
            barra = tk.Frame(marco, bg=color, height=4)
            barra.pack(fill="x")
            titulo = tk.Label(marco, text=MODELOS[clave]["titulo"], bg=PANEL, fg=color, font=(self.f, 11, "bold"), anchor="w")
            titulo.pack(fill="x", padx=10, pady=(8, 0))
            sub = tk.Label(marco, text="", bg=PANEL, fg=SUAVE, font=(self.f, 9), anchor="w", justify="left", wraplength=300)
            sub.pack(fill="x", padx=10)
            caja = scrolledtext.ScrolledText(marco, wrap="word", height=9, font=(self.f, 11), relief="flat",
                                             bg=PANEL, fg=TINTA, padx=10, pady=6, highlightthickness=0)
            caja.pack(fill="both", expand=True)
            caja.config(state="disabled")
            medidas = tk.Label(marco, text="", bg="#f2efe6", fg=TINTA, font=(self.f, 9), anchor="w", justify="left", padx=10, pady=6)
            medidas.pack(fill="x")
            self.col[clave] = {"marco": marco, "sub": sub, "caja": caja, "medidas": medidas}

        # Paso a paso
        pasos = ttk.LabelFrame(p, text="  Cómo eligió cada palabra · haz clic en una fila  ", padding=(8, 6))
        pasos.pack(fill="x", pady=(10, 0))
        fila = ttk.Frame(pasos, style="Panel.TFrame")
        fila.pack(fill="x")
        ttk.Label(fila, text="Ver los pasos de:", style="Suave.TLabel").pack(side="left")
        for valor, texto in (("nuevo", "modelo nuevo"), ("anterior", "modelo anterior")):
            ttk.Radiobutton(fila, text=texto, value=valor, variable=self.modelo_pasos, style="Panel.TRadiobutton",
                            command=self._llenar_pasos).pack(side="left", padx=8)
        self.tabla = ttk.Treeview(pasos, columns=("n", "contexto", "candidatas", "elegida"), show="headings", height=5)
        for col, ancho, titulo in (("n", 30, "#"), ("contexto", 170, "Lo que vio"),
                                   ("candidatas", 250, "Candidatas más probables"), ("elegida", 90, "Eligió")):
            self.tabla.heading(col, text=titulo)
            self.tabla.column(col, width=ancho, minwidth=30 if col == "n" else 60, anchor="w", stretch=(col != "n"))
        self.tabla.pack(fill="x", pady=(6, 0))
        self.tabla.bind("<<TreeviewSelect>>", self._explicar_paso)

        self._ajustar_columnas()

    def _control_deslizante(self, padre, fila, col, texto, variable, minimo, maximo, formato):
        marco = ttk.Frame(padre, style="Panel.TFrame")
        marco.grid(row=fila, column=col, columnspan=2, sticky="we", padx=(0, 14))
        valor = ttk.Label(marco, text=formato.format(variable.get()), style="Panel.TLabel", font=(self.f, 10, "bold"))
        ttk.Label(marco, text=texto, style="Suave.TLabel").pack(side="left")
        valor.pack(side="right")
        ttk.Scale(padre, from_=minimo, to=maximo, variable=variable, orient="horizontal",
                  command=lambda v, e=valor, f=formato: e.config(text=f.format(float(v)))
                  ).grid(row=fila + 1, column=col, columnspan=2, sticky="we", padx=(0, 14))

    def _ajustar_columnas(self):
        e = self.eleccion.get()
        for clave in ("anterior", "nuevo"):
            visible = e == "comparar" or e == clave
            if visible:
                self.col[clave]["marco"].grid()
            else:
                self.col[clave]["marco"].grid_remove()
        if e in ("nuevo", "anterior"):
            self.modelo_pasos.set(e)
        self.cols.columnconfigure(0, weight=1 if e != "nuevo" else 0, uniform="c" if e == "comparar" else "")
        self.cols.columnconfigure(1, weight=1 if e != "anterior" else 0, uniform="c" if e == "comparar" else "")
        if self.modo.get() == "supervisado":
            self._instruccion_inicial_supervisado()

    # ------------------------------------------------------------------ panel no supervisado
    def _construir_panel_no_supervisado(self):
        p = self.panel_uns = tk.Frame(self.zona, bg=FONDO)
        ctrl = ttk.Frame(p, style="Panel.TFrame", padding=12)
        ctrl.pack(fill="x")
        ttk.Label(ctrl, text="Cantidad de grupos (k)", style="Panel.TLabel").grid(row=0, column=0)
        self.k = tk.IntVar(value=3)
        ttk.Spinbox(ctrl, from_=2, to=6, width=4, textvariable=self.k, font=(self.f, 11)).grid(row=0, column=1, padx=8)
        self.boton_agrupar = ttk.Button(ctrl, text="Agrupar canciones", style="Accion.TButton", command=self.agrupar)
        self.boton_agrupar.grid(row=0, column=2, padx=10)
        self.metricas = ttk.Label(ctrl, text="", style="Suave.TLabel")   # segunda fila: no se corta en pantallas chicas
        self.metricas.grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))

        self.zona_grafica = tk.Frame(p, bg=PANEL, highlightbackground=LINEA, highlightthickness=1)
        self.zona_grafica.pack(fill="both", expand=True, pady=(10, 0))
        self.vacio = tk.Label(self.zona_grafica, text="Pulsa «Agrupar canciones» para ver los grupos.",
                              bg=PANEL, fg=SUAVE, font=(self.f, 11))
        self.vacio.pack(expand=True)

        marco = ttk.LabelFrame(p, text="  Grupos encontrados  ", padding=(8, 6))
        marco.pack(fill="x", pady=(10, 0))
        self.tabla_grupos = ttk.Treeview(marco, columns=("g", "n", "comp", "pal"), show="headings", height=4)
        for col, ancho, titulo in (("g", 60, "Grupo"), ("n", 70, "Canciones"), ("comp", 120, "Reales / orig."),
                                   ("pal", 300, "Palabras que más lo distinguen")):
            self.tabla_grupos.heading(col, text=titulo)
            self.tabla_grupos.column(col, width=ancho, minwidth=60, anchor="w", stretch=(col == "pal"))
        self.tabla_grupos.pack(fill="x")
        self.tabla_grupos.bind("<<TreeviewSelect>>", self._explicar_grupo)

    # ------------------------------------------------------------------ cambio de modo
    def cambiar_modo(self):
        self.desplazable.al_inicio()
        if self.modo.get() == "supervisado":
            self.panel_uns.pack_forget()
            self.panel_sup.pack(fill="both", expand=True)
            self._instruccion_inicial_supervisado()
            self.estado.config(text="Modo supervisado: la LSTM aprende de ejemplos con respuesta.")
        else:
            self.panel_sup.pack_forget()
            self.panel_uns.pack(fill="both", expand=True)
            self._instruccion_inicial_no_supervisado()
            self.estado.config(text="Modo no supervisado: el algoritmo agrupa sin respuestas.")

    # ------------------------------------------------------------------ instructor
    def instruir(self, bloques):
        t = self.texto_instructor
        t.config(state="normal")
        t.delete("1.0", "end")
        for estilo, texto in bloques:
            if estilo == "p+":   # parrafo con parte en negrita: lista de (texto, negrita)
                for trozo, negrita in texto:
                    t.insert("end", trozo, ("dato",) if negrita else ("p",))
                t.insert("end", "\n", ("p",))
            else:
                t.insert("end", texto + "\n", (estilo,))
        t.config(state="disabled")
        t.yview_moveto(0)

    def _instruccion_inicial_supervisado(self):
        b = [("h", "Aprendizaje supervisado"),
             ("p", "Aquí el modelo aprende con respuestas. Cada ejemplo es: 6 palabras de una canción → la "
                   "palabra que viene después. Esa palabra siguiente es la «respuesta correcta», por eso se "
                   "llama supervisado."),
             ("p", "Para escribir, la red repite una y otra vez: calcula la probabilidad de cada palabra del "
                   "vocabulario, sortea una, la agrega al texto y vuelve a empezar.")]
        if self.eleccion.get() == "comparar":
            b += [("h", "Qué comparamos"),
                  ("p", "• Modelo anterior: 120 épocas seguidas, sin comprobar nada. Su pérdida llegó a 0.2, "
                        "pero eso era memoria."),
                  ("p", "• Modelo nuevo: se evalúa en cada época con canciones que nunca ve y se frena solo "
                        "cuando deja de mejorar."),
                  ("tip", "Prueba: pulsa «Generar letra» y mira cuánto copia cada modelo de las canciones reales.")]
        else:
            b += [("tip", "Pulsa «Generar letra». Para ver la diferencia entre modelos, elige «Comparar los dos».")]
        self.instruir(b)

    def _instruccion_inicial_no_supervisado(self):
        self.instruir([
            ("h", "Aprendizaje no supervisado"),
            ("p", "Aquí no hay respuestas correctas. Le damos al algoritmo las canciones convertidas en "
                  "números y él solo intenta armar grupos de canciones parecidas (K-Means)."),
            ("h", "Los pasos"),
            ("p", "1. TF-IDF: cuánto «pesa» cada palabra en cada canción.\n2. StandardScaler: poner todo en la misma escala.\n"
                  "3. PCA: reducir dimensiones para poder dibujar en 2D.\n4. K-Means con random_state fijo, para que el resultado se repita.\n"
                  "5. Elegir k con el método del codo y la silueta."),
            ("tip", "Pulsa «Agrupar canciones» y luego cambia k para ver cómo se reordenan los grupos."),
        ])

    # ------------------------------------------------------------------ generacion
    def _lanzar(self, trabajo, al_terminar):
        """Corre 'trabajo' en un hilo (para que la ventana no se congele) y deja el
        resultado en una cola; _revisar_cola lo recoge desde el hilo principal de la ventana."""
        if self.ocupado:
            return
        self.ocupado = True

        def correr():
            try:
                self.cola.put((al_terminar, trabajo(), None))
            except Exception as e:   # noqa: BLE001 - mostramos cualquier error al usuario
                self.cola.put((al_terminar, None, e))
        threading.Thread(target=correr, daemon=True).start()

    def _revisar_cola(self):
        try:
            while True:
                al_terminar, resultado, error = self.cola.get_nowait()
                self._terminar(al_terminar, resultado, error)
        except queue.Empty:
            pass
        self.ventana.after(100, self._revisar_cola)

    def _terminar(self, al_terminar, resultado, error):
        self.ocupado = False
        self.boton_generar.config(state="normal")
        self.boton_agrupar.config(state="normal")
        if error is not None:
            self.estado.config(text="Error. Mira el mensaje.")
            messagebox.showerror("Ocurrió un error", str(error))
            return
        al_terminar(resultado)

    def generar(self):
        semilla = self.entrada_semilla.get().strip()
        if not semilla:
            messagebox.showwarning("Falta la semilla", "Escribe una línea semilla primero.")
            return
        e = self.eleccion.get()
        claves = ["anterior", "nuevo"] if e == "comparar" else [e]
        faltan = [MODELOS[c]["titulo"] for c in claves if not modelo_disponible(c)]
        if faltan:
            messagebox.showwarning("Falta un modelo", "No encuentro el archivo de: " + ", ".join(faltan) +
                                   ".\nEntrena con src/entrenamiento.py o revisa la carpeta models/.")
            return
        temperatura, n = float(self.temperatura.get()), int(self.palabras.get())
        self.boton_generar.config(state="disabled")
        self.estado.config(text="Generando…")

        def trabajo():
            salida = {}
            for c in claves:
                r = generar_con_detalle(semilla, n, temperatura, modelo=c, semilla_azar=sum(map(ord, semilla)))
                r["diagnostico"] = diagnosticar(r["texto"], r["pasos"])
                salida[c] = r
            return salida
        self._lanzar(trabajo, lambda r: self._mostrar_generacion(r, temperatura))

    def _mostrar_generacion(self, resultados, temperatura):
        self.resultados = resultados
        for clave, r in resultados.items():
            c, info, d = self.col[clave], r["info"], r["diagnostico"]
            c["caja"].config(state="normal")
            c["caja"].delete("1.0", "end")
            c["caja"].insert("end", r["texto"])
            c["caja"].config(state="disabled")
            val = f" · pérdida de validación {info['perdida_val']:.2f}" if info["perdida_val"] else " · sin validación"
            c["sub"].config(text=f"{info['epocas']} épocas · {info['parametros']:,} parámetros{val}".replace(",", "."))
            c["medidas"].config(text=f"Copiado {d['pct_copiado']:.0f} %  ·  Seguridad {d['seguridad'] * 100:.0f} %")
        if "nuevo" in resultados:
            os.makedirs(CARPETA_SALIDA_TEXTO, exist_ok=True)
            with open(RUTA_ULTIMA_GENERACION, "w", encoding="utf-8") as f:
                f.write(resultados["nuevo"]["texto"])
            self.boton_escuchar.config(state="normal")
        else:
            self.boton_escuchar.config(state="disabled")
        if self.modelo_pasos.get() not in resultados:
            self.modelo_pasos.set(next(iter(resultados)))
        self._llenar_pasos()
        self.estado.config(text="Letra generada.")
        self._explicar_generacion(temperatura)

    def _explicar_generacion(self, temperatura):
        r = self.resultados
        b = [("h", "Qué estás viendo")]
        desconocidas = sorted({w for x in r.values() for w in x["desconocidas"]})
        if "anterior" in r and "nuevo" in r:
            a, n = r["anterior"]["diagnostico"], r["nuevo"]["diagnostico"]
            b.append(("p+", [("Copiado literal: ", False), (f"el anterior {a['pct_copiado']:.0f} %", True),
                             (" · el nuevo ", False), (f"{n['pct_copiado']:.0f} %", True),
                             (". Es la parte de la letra formada por tramos de 4 palabras seguidas que existen "
                              "tal cual en alguna canción.", False)]))
            if a["pct_copiado"] >= n["pct_copiado"] + 10:
                b.append(("aviso", "El anterior memorizó: entrenó tantas épocas que se aprendió las letras. "
                                   "Suena fluido porque repite frases reales, pero no está creando nada."))
            else:
                b.append(("tip", "En esta tirada los dos copian parecido. Prueba con otra semilla o sube la temperatura."))
            b.append(("p+", [("Seguridad: ", False), (f"{a['seguridad'] * 100:.0f} % vs {n['seguridad'] * 100:.0f} %", True),
                             (". Es la probabilidad media con la que eligió cada palabra. Casi no duda = recuerda; "
                              "duda más = tiene que inventar.", False)]))
            b.append(("p", "Fíjate también en los versos: el nuevo escribe en líneas (marcadas con ¶ en la tabla) porque ahora el entrenamiento "
                           "conserva los saltos de verso; el anterior nunca los vio."))
        else:
            c = next(iter(r))
            d = r[c]["diagnostico"]
            b.append(("p+", [("Copiado literal: ", False), (f"{d['pct_copiado']:.0f} %", True),
                             (" · Seguridad: ", False), (f"{d['seguridad'] * 100:.0f} %", True), (".", False)]))
            b.append(("tip", "Elige «Comparar los dos» para ver el contraste con el otro modelo."))
        b.append(("h", "Pruébalo tú"))
        b.append(("p", f"La temperatura está en {temperatura:.2f}. Bájala (0.4) y el modelo repetirá lo más probable; "
                       "súbela (1.3) y arriesgará más, con más errores."))
        if desconocidas:
            b.append(("aviso", "Palabras de tu semilla que el modelo no conoce: " + ", ".join(desconocidas) +
                               ". Las trata como «?» porque no aparecen en las canciones de entrenamiento."))
        b.append(("aviso", "Ojo: el modelo nuevo todavía escribe mal. Con 75 canciones aprende el «sonido» de las "
                           "letras, pero no el sentido. Lo que más lo mejora es sumar canciones."))
        self.instruir(b)

    # ------------------------------------------------------------------ paso a paso
    def _llenar_pasos(self):
        self.tabla.delete(*self.tabla.get_children())
        r = self.resultados.get(self.modelo_pasos.get())
        if not r:
            return
        for i, p in enumerate(r["pasos"]):
            cand = "  ".join(f"{ver(w)} {pr * 100:.0f}%" for w, pr in p["candidatas"])
            self.tabla.insert("", "end", iid=str(i), values=(
                i + 1, " ".join(ver(w) for w in p["contexto"]), cand, f"{ver(p['elegida'])} ({p['prob_elegida'] * 100:.0f}%)"))

    def _explicar_paso(self, _evento=None):
        sel = self.tabla.selection()
        r = self.resultados.get(self.modelo_pasos.get())
        if not sel or not r:
            return
        p = r["pasos"][int(sel[0])]
        mejor_w, mejor_p = p["candidatas"][0]
        titulo = MODELOS[self.modelo_pasos.get()]["titulo"].lower()
        b = [("h", f"Paso {int(sel[0]) + 1} · {titulo}"),
             ("p", "El modelo vio estas palabras: «" + " ".join(ver(w) for w in p["contexto"]) + "»."),
             ("p+", [("Su favorita era ", False), (f"«{ver(mejor_w)}» con {mejor_p * 100:.0f} %", True), (".", False)])]
        if p["elegida"] == mejor_w:
            b.append(("p", "Y sorteó justo esa."))
        else:
            b.append(("p+", [("Pero sorteó ", False), (f"«{ver(p['elegida'])}» ({p['prob_elegida'] * 100:.0f} %)", True),
                             (". Así se evita repetir siempre lo más probable: es el efecto de la temperatura.", False)]))
        if mejor_p > 0.6:
            b.append(("aviso", "Seguridad muy alta: casi no dudó. En un modelo que memorizó, esto suele significar que "
                               "está recordando una frase que ya vio."))
        elif mejor_p < 0.15:
            b.append(("tip", "Aquí dudó mucho: ninguna palabra destacaba. Es donde el modelo está inventando."))
        b.append(("p", "El símbolo ¶ significa «fin de verso» (salto de línea)."))
        self.instruir(b)

    # ------------------------------------------------------------------ voz
    def escuchar(self):
        if "nuevo" not in self.resultados:
            return
        texto = self.resultados["nuevo"]["texto"]
        self.estado.config(text="Leyendo en voz alta… (la primera vez descarga la voz)")
        self.boton_escuchar.config(state="disabled")

        def trabajo():
            from voz import leer_en_voz_alta
            leer_en_voz_alta(texto, guardar_audio=True)
            leer_en_voz_alta(texto, guardar_audio=False)

        def terminar(_):
            self.estado.config(text="Listo. Audio guardado en outputs/audio/letra_generada.wav")
            self.boton_escuchar.config(state="normal")
        self._lanzar(trabajo, terminar)

    # ------------------------------------------------------------------ agrupamiento
    def agrupar(self):
        try:
            k = int(self.k.get())
        except (tk.TclError, ValueError):
            k = 3
        self.boton_agrupar.config(state="disabled")
        self.estado.config(text="Agrupando…")

        def trabajo():
            from agrupamiento import analizar
            return analizar(k)
        self._lanzar(trabajo, self._mostrar_grupos)

    def _mostrar_grupos(self, r):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.lines import Line2D

        self.resultado_grupos = r
        self.vacio.pack_forget()
        for w in self.zona_grafica.winfo_children():
            w.destroy()

        fig = Figure(figsize=(7.4, 3.3), dpi=100, facecolor=PANEL)
        ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.35, 1]})
        plano, etiquetas, origen = r["plano"], r["etiquetas"], r["origen"]
        for i in range(r["n_canciones"]):
            ax1.scatter(plano[i, 0], plano[i, 1], s=46, color=COLORES_GRUPO[etiquetas[i] % 6],
                        marker="o" if origen[i] == "reales" else "^", edgecolor="white", linewidth=0.7, alpha=0.9)
        ax1.set_title("Canciones agrupadas (PCA en 2D)", fontsize=10, loc="left", color=TINTA)
        ax1.set_xlabel("componente 1", fontsize=8, color=SUAVE)
        ax1.set_ylabel("componente 2", fontsize=8, color=SUAVE)
        leyenda = [Line2D([], [], marker="o", ls="", color=SUAVE, label="real", markersize=6),
                   Line2D([], [], marker="^", ls="", color=SUAVE, label="original", markersize=6)]
        leyenda += [Line2D([], [], marker="s", ls="", color=COLORES_GRUPO[g["id"] % 6], label=f"grupo {g['id']}", markersize=6)
                    for g in r["grupos"]]
        ax1.legend(handles=leyenda, fontsize=7, frameon=False, loc="best")

        ks = r["ks"]
        ax2.plot(ks, r["inercias"], color="#8a8f98", marker="o", lw=2)
        ax2.set_title("Codo y silueta según k", fontsize=10, loc="left", color=TINTA)
        ax2.set_xlabel("k (cantidad de grupos)", fontsize=8, color=SUAVE)
        ax2.set_ylabel("inercia (más baja = grupos más compactos)", fontsize=7, color="#8a8f98")
        ax3 = ax2.twinx()
        ax3.plot(ks, r["siluetas"], color="#d9822b", marker="s", lw=2)
        ax3.set_ylabel("silueta (más alta = mejor separados)", fontsize=7, color="#d9822b")
        ax2.axvline(r["k"], color=ACENTO, ls="--", lw=1.2)
        for ax in (ax1, ax2, ax3):
            ax.tick_params(labelsize=7, colors=SUAVE)
            for lado in ("top",):
                ax.spines[lado].set_visible(False)
        ax2.set_xticks(ks)
        fig.tight_layout()
        self.fig = fig
        canvas = FigureCanvasTkAgg(fig, master=self.zona_grafica)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

        self.metricas.config(text=f"Silueta {r['silueta']:.2f}   ·   Rand ajustado {r['rand_ajustado']:.2f}   ·   Info. mutua {r['info_mutua']:.2f}")
        self.tabla_grupos.delete(*self.tabla_grupos.get_children())
        for g in r["grupos"]:
            comp = f"{g['composicion'].get('reales', 0)} / {g['composicion'].get('originales', 0)}"
            self.tabla_grupos.insert("", "end", iid=str(g["id"]), values=(f"Grupo {g['id']}", g["tamano"], comp, ", ".join(g["palabras"])))
        self.estado.config(text=f"Listo: {r['n_canciones']} canciones en {r['k']} grupos.")
        self._explicar_agrupamiento(r)

    def _explicar_agrupamiento(self, r):
        pequenos = [g for g in r["grupos"] if g["tamano"] <= 3]
        b = [("h", "Qué estás viendo"),
             ("p", f"El algoritmo repartió {r['n_canciones']} canciones en {r['k']} grupos sin que nadie le dijera cuáles."),
             ("p+", [("Silueta ", False), (f"{r['silueta']:.2f}", True),
                     (": va de -1 a 1; cerca de 1 = grupos bien separados, cerca de 0 = se mezclan.", False)])]
        if pequenos:
            nombres = "; ".join(n.replace("_-_", " – ") for g in pequenos for n in g["ejemplos"])
            b.append(("aviso", f"Hay un grupo de solo {pequenos[0]['tamano']} canciones: {nombres}. Suelen ser canciones "
                               "atípicas (por ejemplo, con partes de la letra en inglés). Una silueta alta con grupos así engaña: "
                               "K-Means solo separó lo raro, no encontró estilos."))
        mejor = r["ks"][max(range(len(r["siluetas"])), key=r["siluetas"].__getitem__)]
        b.append(("p", f"La mejor silueta está en k = {mejor}. El codo (línea gris) no muestra un quiebre claro, "
                       "señal de que los datos no se separan en grupos nítidos."))
        b.append(("p+", [("Rand ajustado ", False), (f"{r['rand_ajustado']:.2f}", True), (" e información mutua ", False),
                         (f"{r['info_mutua']:.2f}", True),
                         (": comparé los grupos con algo que sí conocemos, si la canción es de un artista real "
                          "(círculo en el gráfico) o escrita para el proyecto (triángulo). Cerca de 0 = los grupos no tienen relación con eso. "
                          "No es un fallo: ese origen no es un estilo.", False)]))
        b.append(("tip", "Haz clic en un grupo de la tabla para ver ejemplos. Prueba k = 2, 3 y 4 y compara."))
        self.instruir(b)

    def _explicar_grupo(self, _evento=None):
        sel = self.tabla_grupos.selection()
        if not sel or not self.resultado_grupos:
            return
        g = self.resultado_grupos["grupos"][int(sel[0])]
        self.instruir([
            ("h", f"Grupo {g['id']} · {g['tamano']} canciones"),
            ("p", "Palabras que más lo distinguen de las demás: " + ", ".join(g["palabras"]) + "."),
            ("p", "Ejemplos: " + "; ".join(n.replace("_-_", " – ") for n in g["ejemplos"]) + "."),
            ("tip", "Mira las palabras y decide tú si el grupo tiene sentido (¿habla de baile? ¿de una persona? ¿de un lugar?). "
                    "Poner nombre a un grupo es trabajo humano: el algoritmo no sabe de qué hablan."),
        ])


def main():
    ventana = tk.Tk()
    App(ventana)
    ventana.mainloop()


if __name__ == "__main__":
    main()
