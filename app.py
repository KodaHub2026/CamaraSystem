"""
app.py
======
Panel de control de Vigilancia Emocional.

    python app.py

La camara NO se abre al arrancar. Primero configuras fuente, modelo y
parametros; el analisis empieza cuando presionas INICIAR.

Sobre Tkinter: se eligio porque viene incluido con Python en Windows. Cero
instalacion extra, cero dependencias pesadas, y arranca al instante. El video
se pinta dentro de la ventana con Pillow, escalado para no competir con
MediaPipe por CPU: se ANALIZA a resolucion completa (hacen falta pixeles de
rostro) y se MUESTRA reducido.

Regla de oro del hilo principal: el motor corre aparte y deja frames en una
ranura; esta ventana los recoge cada 33 ms con `after()`. Nunca se llama a un
widget desde otro hilo.
"""

from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from src.camaras import Camara, detectar_async
from src.catalogo import CatalogoModelos
from src.llm import URL_POR_DEFECTO
from src.motor import Configuracion, MotorVigilancia

RAIZ = Path(__file__).parent

# Paleta KodaHub
FONDO = "#1A1F2B"
PANEL = "#222836"
BORDE = "#323A4D"
TEXTO = "#E8ECF2"
TENUE = "#8A93A6"
LIMA = "#8DC63F"
CIAN = "#00D9E8"
ROJO = "#E05252"
AMBAR = "#F0B429"


class Aplicacion(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Vigilancia Emocional — KodaHub")
        self.geometry("1180x720")
        self.minsize(1040, 640)
        self.configure(bg=FONDO)

        self.motor: MotorVigilancia | None = None
        self.catalogo = CatalogoModelos(URL_POR_DEFECTO)
        self._foto = None                 # referencia viva: si se libera, la imagen desaparece
        self._eventos: queue.Queue = queue.Queue()

        self._estilos()
        self._construir()
        self.protocol("WM_DELETE_WINDOW", self._cerrar)
        self.after(33, self._tick)
        self._refrescar_modelos()
        self._buscar_camaras()

    # ------------------------------------------------------------------ #
    # Apariencia
    # ------------------------------------------------------------------ #

    def _estilos(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=PANEL, foreground=TEXTO, borderwidth=0)
        s.configure("TFrame", background=PANEL)
        s.configure("Fondo.TFrame", background=FONDO)
        s.configure("TLabel", background=PANEL, foreground=TEXTO, font=("Segoe UI", 9))
        s.configure("Titulo.TLabel", font=("Segoe UI", 8, "bold"), foreground=TENUE)
        s.configure("Dato.TLabel", font=("Segoe UI", 15, "bold"), foreground=TEXTO)
        s.configure("Tenue.TLabel", foreground=TENUE, font=("Segoe UI", 8))
        s.configure("TRadiobutton", background=PANEL, foreground=TEXTO, font=("Segoe UI", 9))
        s.map("TRadiobutton", background=[("active", PANEL)])
        s.configure("TCheckbutton", background=PANEL, foreground=TEXTO, font=("Segoe UI", 9))
        s.map("TCheckbutton", background=[("active", PANEL)])
        s.configure("TEntry", fieldbackground=FONDO, foreground=TEXTO,
                    insertcolor=TEXTO, bordercolor=BORDE)
        s.configure("TCombobox", fieldbackground=FONDO, background=FONDO,
                    foreground=TEXTO, arrowcolor=TENUE, bordercolor=BORDE)
        s.map("TCombobox", fieldbackground=[("readonly", FONDO)])
        s.configure("TSpinbox", fieldbackground=FONDO, foreground=TEXTO,
                    arrowcolor=TENUE, bordercolor=BORDE)
        s.configure("Accion.TButton", background=LIMA, foreground="#14181F",
                    font=("Segoe UI", 10, "bold"), padding=9)
        s.map("Accion.TButton", background=[("active", "#A0DC50"), ("disabled", BORDE)])
        s.configure("Alto.TButton", background=ROJO, foreground="#FFFFFF",
                    font=("Segoe UI", 10, "bold"), padding=9)
        s.map("Alto.TButton", background=[("active", "#F06565")])
        s.configure("Sec.TButton", background=BORDE, foreground=TEXTO,
                    font=("Segoe UI", 8), padding=5)
        s.map("Sec.TButton", background=[("active", "#3F4863")])

    def _seccion(self, padre, titulo) -> ttk.Frame:
        ttk.Label(padre, text=titulo, style="Titulo.TLabel").pack(anchor="w", pady=(12, 4))
        f = ttk.Frame(padre)
        f.pack(fill="x")
        return f

    # ------------------------------------------------------------------ #
    # Construccion
    # ------------------------------------------------------------------ #

    def _construir(self) -> None:
        cont = ttk.Frame(self, style="Fondo.TFrame")
        cont.pack(fill="both", expand=True, padx=10, pady=10)

        izq = ttk.Frame(cont, width=290)
        izq.pack(side="left", fill="y", padx=(0, 10))
        izq.pack_propagate(False)
        self._panel_izquierdo(izq)

        der = ttk.Frame(cont, style="Fondo.TFrame")
        der.pack(side="left", fill="both", expand=True)
        self._panel_derecho(der)

    # ----------------------------- izquierda -------------------------- #

    def _panel_izquierdo(self, p) -> None:
        interior = ttk.Frame(p)
        interior.pack(fill="both", expand=True, padx=12, pady=8)

        ttk.Label(interior, text="VIGILANCIA EMOCIONAL",
                  font=("Segoe UI", 11, "bold"), foreground=CIAN).pack(anchor="w")
        ttk.Label(interior, text="KodaHub", style="Tenue.TLabel").pack(anchor="w")

        # --- Fuente ---
        f = self._seccion(interior, "FUENTE DE VIDEO")
        self.var_fuente = tk.StringVar(value="usb")
        for val, txt in (("usb", "Camara USB"), ("rtsp", "Camara IP (RTSP)"),
                         ("archivo", "Archivo de video")):
            ttk.Radiobutton(f, text=txt, value=val, variable=self.var_fuente,
                            command=self._cambio_fuente).pack(anchor="w")

        # Contenedor FIJO. Los marcos dinamicos viven adentro, asi que
        # pack/pack_forget entre ellos no reordena el resto del panel.
        # (Bug corregido: sin esto, cada cambio de fuente mandaba el bloque
        #  al fondo de la ventana, lejos del radio que lo controla.)
        self.cont_fuente = ttk.Frame(interior)
        self.cont_fuente.pack(fill="x", pady=(4, 0))

        self.marco_usb = ttk.Frame(self.cont_fuente)
        fila_cam = ttk.Frame(self.marco_usb)
        fila_cam.pack(fill="x")
        self.var_camara = tk.StringVar(value="(buscar camaras)")
        self.combo_cam = ttk.Combobox(fila_cam, textvariable=self.var_camara,
                                      state="readonly", values=[])
        self.combo_cam.pack(side="left", fill="x", expand=True)
        self.btn_buscar = ttk.Button(fila_cam, text="⟳", width=3, style="Sec.TButton",
                                     command=self._buscar_camaras)
        self.btn_buscar.pack(side="left", padx=(4, 0))
        self.lbl_cam = ttk.Label(self.marco_usb, text="Presiona ⟳ para detectar",
                                 style="Tenue.TLabel")
        self.lbl_cam.pack(anchor="w", pady=(3, 0))
        self._camaras: list[Camara] = []

        self.marco_rtsp = ttk.Frame(self.cont_fuente)
        self.var_rtsp = tk.StringVar(value="rtsp://usuario:clave@10.0.0.50:554/Streaming/Channels/101")
        ttk.Entry(self.marco_rtsp, textvariable=self.var_rtsp).pack(fill="x")

        self.marco_archivo = ttk.Frame(self.cont_fuente)
        self.var_archivo = tk.StringVar()
        ttk.Entry(self.marco_archivo, textvariable=self.var_archivo).pack(
            side="left", fill="x", expand=True)
        ttk.Button(self.marco_archivo, text="...", width=3, style="Sec.TButton",
                   command=self._elegir_archivo).pack(side="left", padx=(4, 0))

        self._cambio_fuente()

        # --- Modelo de IA ---
        f = self._seccion(interior, "MODELO DE IA")
        self.var_llm = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Activar interpretacion", variable=self.var_llm,
                        command=self._cambio_llm).pack(anchor="w")

        self.cont_llm = ttk.Frame(interior)
        self.cont_llm.pack(fill="x")
        self.marco_llm = ttk.Frame(self.cont_llm)
        ttk.Label(self.marco_llm, text="Servidor", style="Tenue.TLabel").pack(anchor="w")
        self.var_url = tk.StringVar(value=URL_POR_DEFECTO)
        ttk.Entry(self.marco_llm, textvariable=self.var_url).pack(fill="x", pady=(0, 6))

        fila = ttk.Frame(self.marco_llm)
        fila.pack(fill="x")
        self.var_modelo = tk.StringVar()
        self.combo = ttk.Combobox(fila, textvariable=self.var_modelo,
                                  state="readonly", values=[])
        self.combo.pack(side="left", fill="x", expand=True)
        ttk.Button(fila, text="↻", width=3, style="Sec.TButton",
                   command=self._refrescar_modelos).pack(side="left", padx=(4, 0))

        fila2 = ttk.Frame(self.marco_llm)
        fila2.pack(fill="x", pady=(6, 0))
        self.btn_probar = ttk.Button(fila2, text="Probar modelo", style="Sec.TButton",
                                     command=self._probar_modelo)
        self.btn_probar.pack(side="left")
        self.lbl_prueba = ttk.Label(fila2, text="sin probar", style="Tenue.TLabel")
        self.lbl_prueba.pack(side="left", padx=8)

        ttk.Label(self.marco_llm,
                  text="Aparecer en la lista no significa\nestar cargado. Pruebalo antes.",
                  style="Tenue.TLabel", justify="left").pack(anchor="w", pady=(6, 0))

        # --- Parametros ---
        f = self._seccion(interior, "PARAMETROS")
        self.var_aforo = tk.IntVar(value=6)
        self.var_umbral = tk.DoubleVar(value=0.22)
        self.var_saltar = tk.IntVar(value=1)
        for etiqueta, var, desde, hasta, paso in (
            ("Aforo maximo", self.var_aforo, 1, 20, 1),
            ("Umbral", self.var_umbral, 0.04, 0.9, 0.02),
            ("Analizar 1 de cada", self.var_saltar, 1, 6, 1),
        ):
            fila = ttk.Frame(f)
            fila.pack(fill="x", pady=2)
            ttk.Label(fila, text=etiqueta, style="Tenue.TLabel").pack(side="left")
            ttk.Spinbox(fila, from_=desde, to=hasta, increment=paso, width=6,
                        textvariable=var).pack(side="right")

        self.var_csv = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Guardar metricas a CSV",
                        variable=self.var_csv).pack(anchor="w", pady=(6, 0))

        # --- Boton principal ---
        self.btn_inicio = ttk.Button(interior, text="INICIAR", style="Accion.TButton",
                                     command=self._alternar)
        self.btn_inicio.pack(fill="x", side="bottom", pady=(12, 0))

        self.lbl_error = ttk.Label(interior, text="", style="Tenue.TLabel",
                                   foreground=ROJO, wraplength=260, justify="left")
        self.lbl_error.pack(fill="x", side="bottom", pady=(0, 6))

    # ----------------------------- derecha ---------------------------- #

    def _panel_derecho(self, p) -> None:
        self.lienzo = tk.Canvas(p, bg="#12161F", highlightthickness=1,
                                highlightbackground=BORDE)
        self.lienzo.pack(fill="both", expand=True)
        self.lienzo.create_text(
            10, 10, anchor="nw", tags="hint", fill=TENUE, font=("Segoe UI", 10),
            text="La camara esta apagada.\n\nConfigura la fuente y presiona INICIAR.",
        )

        barra = ttk.Frame(p)
        barra.pack(fill="x", pady=(10, 0))
        self.indicadores = {}
        for clave, etiqueta in (("aforo", "AFORO"), ("fiables", "FIABLES"),
                                ("calidad", "CALIDAD"), ("dominante", "PREDOMINANTE"),
                                ("descartadas", "DESCARTADAS"), ("fps", "FPS")):
            celda = ttk.Frame(barra)
            celda.pack(side="left", expand=True, fill="x", padx=3)
            ttk.Label(celda, text=etiqueta, style="Titulo.TLabel").pack(anchor="w")
            lbl = ttk.Label(celda, text="—", style="Dato.TLabel")
            lbl.pack(anchor="w")
            self.indicadores[clave] = lbl

        inferior = ttk.Frame(p)
        inferior.pack(fill="x", pady=(10, 0))

        acc = ttk.Frame(inferior)
        acc.pack(side="left", fill="y", padx=(0, 10))
        self.var_malla = tk.BooleanVar(value=False)
        self.var_calidad = tk.BooleanVar(value=False)
        ttk.Checkbutton(acc, text="Malla facial", variable=self.var_malla,
                        command=self._sync_dibujo).pack(anchor="w")
        ttk.Checkbutton(acc, text="Detalle de calidad", variable=self.var_calidad,
                        command=self._sync_dibujo).pack(anchor="w")
        self.btn_interpretar = ttk.Button(acc, text="Interpretar ahora", style="Sec.TButton",
                                          command=self._interpretar, state="disabled")
        self.btn_interpretar.pack(anchor="w", pady=(6, 0))

        caja = ttk.Frame(inferior)
        caja.pack(side="left", fill="both", expand=True)
        ttk.Label(caja, text="INTERPRETACION", style="Titulo.TLabel").pack(anchor="w")
        self.lbl_llm = ttk.Label(caja, text="Interpretacion desactivada.",
                                 style="Tenue.TLabel", wraplength=560, justify="left")
        self.lbl_llm.pack(anchor="w", fill="x")

    # ------------------------------------------------------------------ #
    # Interacciones
    # ------------------------------------------------------------------ #

    def _cambio_fuente(self) -> None:
        for m in (self.marco_usb, self.marco_rtsp, self.marco_archivo):
            m.pack_forget()
        {"usb": self.marco_usb, "rtsp": self.marco_rtsp,
         "archivo": self.marco_archivo}[self.var_fuente.get()].pack(fill="x", pady=(4, 0))

    def _cambio_llm(self) -> None:
        if self.var_llm.get():
            self.marco_llm.pack(fill="x", pady=(6, 0))
        else:
            self.marco_llm.pack_forget()

    def _elegir_archivo(self) -> None:
        ruta = filedialog.askopenfilename(
            title="Elegir video",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov"), ("Todos", "*.*")],
        )
        if ruta:
            self.var_archivo.set(ruta)

    def _sync_dibujo(self) -> None:
        if self.motor:
            self.motor.config.mostrar_malla = self.var_malla.get()
            self.motor.config.mostrar_calidad = self.var_calidad.get()

    def _buscar_camaras(self) -> None:
        self.btn_buscar.configure(state="disabled")
        self.lbl_cam.configure(text="buscando dispositivos...", foreground=AMBAR)
        self.combo_cam.configure(values=[])
        self.var_camara.set("buscando...")
        detectar_async(lambda cams: self._eventos.put(("camaras", cams, None)))

    def _refrescar_modelos(self) -> None:
        self.catalogo.base_url = self.var_url.get().strip() or URL_POR_DEFECTO
        self.combo.configure(values=["consultando..."])
        self.var_modelo.set("consultando...")
        self.catalogo.listar_async(
            lambda mods, err: self._eventos.put(("modelos", mods, err))
        )

    def _probar_modelo(self) -> None:
        modelo = self.var_modelo.get()
        if not modelo or modelo.startswith("("):
            return
        self.btn_probar.configure(state="disabled")
        self.lbl_prueba.configure(text="probando...", foreground=AMBAR)
        self.catalogo.probar_async(
            modelo, lambda res: self._eventos.put(("prueba", res, None))
        )

    def _interpretar(self) -> None:
        if self.motor and self.motor.solicitar_interpretacion():
            self.lbl_llm.configure(text="Consultando al modelo...", foreground=AMBAR)

    # ------------------------------------------------------------------ #

    def _construir_config(self) -> Configuracion:
        tipo = self.var_fuente.get()
        if tipo == "usb":
            cam = next((c for c in self._camaras
                        if c.etiqueta == self.var_camara.get()), None)
            if cam is None:
                raise ValueError("Presiona ⟳ y elige una camara de la lista")
            origen, backend = cam.indice, cam.backend
        elif tipo == "rtsp":
            origen = self.var_rtsp.get().strip()
            backend = None
            if not origen.lower().startswith("rtsp"):
                raise ValueError("La URL debe empezar con rtsp://")
            backend = None
        else:
            origen = self.var_archivo.get().strip()
            backend = None
            if not origen or not Path(origen).exists():
                raise ValueError("Elige un archivo de video valido")

        csv = None
        if self.var_csv.get():
            carpeta = RAIZ / "registros"
            carpeta.mkdir(exist_ok=True)
            from datetime import datetime
            csv = str(carpeta / f"sesion_{datetime.now():%Y%m%d_%H%M%S}.csv")

        return Configuracion(
            origen=origen, tipo=tipo, backend=backend,
            aforo=self.var_aforo.get(), umbral=float(self.var_umbral.get()),
            saltar=max(1, self.var_saltar.get()),
            llm_activo=self.var_llm.get(),
            llm_url=self.var_url.get().strip(),
            llm_modelo=self.var_modelo.get() if self.var_llm.get() else "",
            salida_csv=csv,
            mostrar_malla=self.var_malla.get(),
            mostrar_calidad=self.var_calidad.get(),
        )

    def _alternar(self) -> None:
        if self.motor is not None and self.motor.estado.corriendo:
            self._detener()
        else:
            self._iniciar()

    def _iniciar(self) -> None:
        try:
            cfg = self._construir_config()
        except ValueError as e:
            self.lbl_error.configure(text=str(e), foreground=ROJO)
            return

        # Matar SIEMPRE el motor anterior antes de crear otro. Si el motor
        # murio por un error, self.motor sigue apuntando a el con su hilo
        # posiblemente vivo y la camara tomada. Crear uno encima dejaba dos
        # hilos peleando por el mismo dispositivo y por MediaPipe, lo que
        # revienta en C con una violacion de acceso (0x101), no con una
        # excepcion de Python.
        if self.motor is not None:
            self.motor.detener(esperar=2.5)
            self.motor = None

        self.lbl_error.configure(text="")
        self.motor = MotorVigilancia(cfg)
        self.motor.iniciar()
        self.btn_inicio.configure(text="DETENER", style="Alto.TButton", state="disabled")
        # Rearmar tras un momento: evita que un doble clic rapido dispare
        # dos arranques antes de que el primero registre su estado.
        self.after(900, lambda: self.btn_inicio.configure(state="normal"))
        self.btn_interpretar.configure(state="normal" if cfg.llm_activo else "disabled")
        self.lienzo.delete("hint")

    def _detener(self, mostrar_resumen: bool = True) -> None:
        if not self.motor:
            return
        resumen = self.motor.resumen_final()
        self.motor.detener()
        self.motor = None
        self.btn_inicio.configure(text="INICIAR", style="Accion.TButton")
        self.btn_interpretar.configure(state="disabled")

        if mostrar_resumen and resumen.get("total"):
            pct = resumen["descartadas_pct"]
            aviso = ""
            if pct > 45:
                aviso = ("\n\nMas del 45% descartado. Revisa altura, angulo e "
                         "iluminacion antes de dar peso a estos numeros.")
            messagebox.showinfo(
                "Resumen de sesion",
                f"Lecturas: {resumen['total']}\n"
                f"Descartadas por calidad: {pct:.1f}%\n"
                f"Pico de aforo: {resumen['pico_aforo']}\n"
                f"Predominante: {resumen['dominante']}\n"
                f"Calidad media: {resumen['calidad_media']:.2f}{aviso}",
            )

    # ------------------------------------------------------------------ #
    # Ciclo de refresco (hilo principal)
    # ------------------------------------------------------------------ #

    def _tick(self) -> None:
        self._drenar_eventos()
        if self.motor:
            self._pintar_video()
            self._pintar_estado()
        self.after(33, self._tick)

    def _drenar_eventos(self) -> None:
        """Resultados de hilos secundarios llegan por cola y se aplican aqui."""
        while True:
            try:
                tipo, dato, extra = self._eventos.get_nowait()
            except queue.Empty:
                return

            if tipo == "modelos":
                if extra:
                    self.combo.configure(values=[])
                    self.var_modelo.set("")
                    self.lbl_prueba.configure(text=extra, foreground=ROJO)
                else:
                    self.combo.configure(values=dato)
                    if dato:
                        self.var_modelo.set(dato[0])
                    self.lbl_prueba.configure(
                        text=f"{len(dato)} modelos", foreground=TENUE)

            elif tipo == "camaras":
                self.btn_buscar.configure(state="normal")
                self._camaras = dato
                if not dato:
                    self.combo_cam.configure(values=[])
                    self.var_camara.set("")
                    self.lbl_cam.configure(
                        text="Ninguna camara respondio.\nCierrala en Teams, Zoom u OBS.",
                        foreground=ROJO)
                else:
                    etiquetas = [c.etiqueta for c in dato]
                    self.combo_cam.configure(values=etiquetas)
                    self.var_camara.set(etiquetas[0])
                    self.lbl_cam.configure(
                        text=f"{len(dato)} disponible(s)", foreground=LIMA)

            elif tipo == "prueba":
                self.btn_probar.configure(state="normal")
                self.lbl_prueba.configure(text=dato.resumen, foreground=dato.color)

    def _pintar_video(self) -> None:
        frame = self.motor.tomar_frame()
        if frame is None:
            return

        cw = max(self.lienzo.winfo_width(), 1)
        ch = max(self.lienzo.winfo_height(), 1)
        if cw < 20 or ch < 20:
            return

        h, w = frame.shape[:2]
        escala = min(cw / w, ch / h)
        nw, nh = int(w * escala), int(h * escala)

        # Se muestra reducido pero se ANALIZO a resolucion completa.
        pequeno = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(pequeno, cv2.COLOR_BGR2RGB)

        self._foto = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.lienzo.delete("video")
        self.lienzo.create_image(cw // 2, ch // 2, image=self._foto,
                                 anchor="center", tags="video")

    def _pintar_estado(self) -> None:
        e = self.motor.estado

        # El motor puede detenerse solo (archivo terminado o error de fuente).
        # OJO: hay que RESCATAR el error ANTES de llamar a _detener(), porque
        # _detener() destruye el motor y con el su estado. Sin esto el mensaje
        # se perdia y el usuario solo veia "FPS 0" sin ninguna explicacion.
        if e.arrancando:
            self.lbl_error.configure(text="Abriendo la camara...", foreground=AMBAR)
            return

        if not e.corriendo and self.btn_inicio["text"] == "DETENER":
            error_rescatado = e.error
            termino = e.termino_solo
            self._detener(mostrar_resumen=not error_rescatado)
            if error_rescatado:
                self.lbl_error.configure(text=error_rescatado, foreground=ROJO)
                print(f"[error] {error_rescatado}", flush=True)
            elif termino:
                self.lbl_error.configure(text="Video terminado.", foreground=TENUE)
            return

        if self.lbl_error["text"] == "Abriendo la camara...":
            self.lbl_error.configure(text="")

        if e.error:
            self.lbl_error.configure(text=e.error, foreground=ROJO)

        q = e.calidad_media
        self.indicadores["aforo"].configure(text=str(e.aforo))
        self.indicadores["fiables"].configure(text=str(e.fiables))
        self.indicadores["calidad"].configure(
            text=f"{q:.2f}",
            foreground=LIMA if q > 0.6 else (AMBAR if q > 0.35 else ROJO))
        self.indicadores["dominante"].configure(text=e.dominante[:11])
        self.indicadores["descartadas"].configure(
            text=f"{e.descartadas_pct:.0f}%",
            foreground=ROJO if e.descartadas_pct > 45 else TEXTO)
        self.indicadores["fps"].configure(text=f"{e.fps:.0f}")

        if self.motor.interprete:
            u = self.motor.interprete.ultima
            if self.motor.interprete.ocupado:
                self.lbl_llm.configure(text="Consultando al modelo...", foreground=AMBAR)
            elif u and not u.error:
                txt = f"{u.matiz.upper()}  —  {u.texto}" if u.matiz else u.texto
                self.lbl_llm.configure(text=f"{txt}\n({u.hora}, {u.segundos:.0f}s)",
                                       foreground=TEXTO)
            elif u:
                self.lbl_llm.configure(text=u.texto, foreground=ROJO)

    def _cerrar(self) -> None:
        if self.motor:
            self.motor.detener(esperar=2.0)
        self.destroy()


if __name__ == "__main__":
    Aplicacion().mainloop()
