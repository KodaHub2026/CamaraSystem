"""
motor.py
========
Motor de captura y analisis, corriendo en su propio hilo.

POR QUE ESTA SEPARADO DE LA INTERFAZ
------------------------------------
Tkinter no es thread-safe: toda actualizacion de widgets tiene que ocurrir en el
hilo principal. Y el analisis (captura + MediaPipe + seguimiento) consume entre
20 y 35 ms por frame. Si eso corriera en el hilo de la GUI, la ventana quedaria
congelada: los botones no responderian y Windows la marcaria como "no responde".

El patron es el estandar para video en Tkinter:

    hilo motor  ->  captura, analiza, deja el frame en una ranura protegida
    hilo GUI    ->  cada 33 ms lee la ranura y pinta

El motor no sabe que existe una interfaz. Expone estado y frames; quien lo
consuma decide como mostrarlos. Por eso el mismo motor sirve para la GUI, para
el modo desatendido y para cualquier cosa que venga despues.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .agregado import Agregador
from .detector import DetectorEmociones
from .fuente import FuenteVideo
from .llm import BufferTemporal, ClienteLMStudio, InterpreteAsincrono
from .seguimiento import Seguidor
from .ui import dibujar_hud_vigilancia, dibujar_rostro_vigilancia

RAIZ = Path(__file__).parent.parent


@dataclass
class Configuracion:
    """Todo lo que la interfaz puede cambiar antes de arrancar."""

    origen: object = 0                  # indice int, url rtsp o ruta de archivo
    tipo: str = "usb"                   # usb | rtsp | archivo
    ancho: int = 1280
    alto: int = 720
    aforo: int = 6
    umbral: float = 0.22
    suavizado: int = 10
    saltar: int = 1
    espejo: bool = True
    backend: int | None = None      # backend de camara ya validado por camaras.detectar

    llm_activo: bool = False
    llm_url: str = ""
    llm_modelo: str = ""
    llm_auto: float = 0.0

    salida_csv: str | None = None
    alerta_segundos: float = 4.0
    alerta_aforo: int = 0

    # Opciones de dibujo, conmutables en caliente
    mostrar_malla: bool = False
    mostrar_calidad: bool = False
    mostrar_hud: bool = True


@dataclass
class Estado:
    """Instantanea que la interfaz lee para pintar sus indicadores."""

    corriendo: bool = False
    conectado: bool = False
    arrancando: bool = False     # abriendo la fuente; aun no hay frames
    termino_solo: bool = False   # el archivo llego al final sin que nadie parara
    fps: float = 0.0
    aforo: int = 0
    fiables: int = 0
    calidad_media: float = 0.0
    dominante: str = "-"
    descartadas_pct: float = 0.0
    reconexiones: int = 0
    transcurrido: float = 0.0
    error: str = ""
    alertas: list = field(default_factory=list)


class MotorVigilancia:
    """
    Captura y analiza en segundo plano.

    Uso:
        motor = MotorVigilancia(config)
        motor.iniciar()
        frame = motor.tomar_frame()     # None si aun no hay
        estado = motor.estado           # instantanea segura
        motor.detener()
    """

    def __init__(self, config: Configuracion) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._estado = Estado()
        self._hilo: threading.Thread | None = None
        self._parar = threading.Event()

        self.agregador: Agregador | None = None
        self.interprete: InterpreteAsincrono | None = None
        self.buffer = BufferTemporal(ventana_seg=30.0)

    # ------------------------------------------------------------------ #
    # API publica
    # ------------------------------------------------------------------ #

    @property
    def estado(self) -> Estado:
        with self._lock:
            return self._estado

    def tomar_frame(self) -> np.ndarray | None:
        """Devuelve una COPIA del ultimo frame analizado, o None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def iniciar(self) -> None:
        if self._hilo and self._hilo.is_alive():
            return
        self._parar.clear()

        # CRITICO: marcar corriendo=True AQUI, en el hilo que llama, antes de
        # lanzar el worker. Abrir una camara con DirectShow tarda entre 1 y 3
        # segundos; si dejamos corriendo=False hasta que el hilo llegue a
        # ponerlo, cualquier consumidor que revise el estado en ese intervalo
        # cree que el motor ya murio.
        #
        # Ese fue el bug: la interfaz revisa cada 33 ms, veia corriendo=False
        # 33 ms despues de arrancar, y mataba el motor mientras la camara
        # apenas estaba abriendo. El LED prendia y se apagaba enseguida.
        #
        # El worker baja la bandera si falla o al terminar.
        with self._lock:
            self._estado = Estado(corriendo=True, arrancando=True)
            self._frame = None

        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def detener(self, esperar: float = 3.0) -> None:
        self._parar.set()
        if self._hilo:
            self._hilo.join(timeout=esperar)
        self._hilo = None
        with self._lock:
            self._estado.corriendo = False

    def solicitar_interpretacion(self) -> bool:
        """Pide a la IA que interprete el momento actual. No bloquea."""
        if not self.interprete:
            return False
        return self.interprete.solicitar_desde_buffer(self.buffer)

    def solicitar_resumen(self) -> bool:
        if not self.interprete:
            return False
        return self.interprete.solicitar_sesion(self.buffer, self.estado.transcurrido)

    def resumen_final(self) -> dict:
        if not self.agregador:
            return {}
        r = self.agregador.resumen_ventana()
        r["descartadas_pct"] = self._pct_descartadas()
        r["pico_aforo"] = self.agregador.pico_aforo
        r["total"] = self.agregador.total_lecturas
        return r

    # ------------------------------------------------------------------ #
    # Bucle interno
    # ------------------------------------------------------------------ #

    def _pct_descartadas(self) -> float:
        if not self.agregador or not self.agregador.total_lecturas:
            return 0.0
        return self.agregador.descartadas_calidad / self.agregador.total_lecturas * 100

    def _bucle(self) -> None:
        cfg = self.config
        self.agregador = Agregador(
            salida=cfg.salida_csv,
            alerta_segundos=cfg.alerta_segundos,
            alerta_aforo=cfg.alerta_aforo,
        )
        seguidor = Seguidor()

        if cfg.llm_activo and cfg.llm_url and cfg.llm_modelo:
            self.interprete = InterpreteAsincrono(
                ClienteLMStudio(base_url=cfg.llm_url, modelo=cfg.llm_modelo)
            )

        fuente = FuenteVideo(cfg.origen, cfg.ancho, cfg.alto, backend=cfg.backend)
        if not fuente.abrir():
            with self._lock:
                self._estado.error = (
                    f"No se pudo abrir la fuente: {cfg.origen}\n"
                    f"{fuente.estado.ultimo_error}"
                )
                self._estado.corriendo = False
                self._estado.arrancando = False
            return

        with self._lock:
            self._estado.corriendo = True
            self._estado.conectado = True
            self._estado.arrancando = False   # fuente abierta; ya vienen frames
            self._estado.error = ""

        # Crear el detector puede fallar (modelo ausente, API incompatible,
        # falta de memoria). Si truena aqui, hay que soltar la camara: si no,
        # queda tomada y el siguiente intento tampoco puede abrirla.
        try:
            detector = DetectorEmociones(
                ruta_modelo=RAIZ / "modelos" / "face_landmarker.task",
                umbral=cfg.umbral,
                num_rostros=cfg.aforo,
                ventana_suavizado=cfg.suavizado,
            )
        except Exception as e:  # noqa: BLE001
            fuente.cerrar()
            with self._lock:
                self._estado.error = f"No se pudo crear el detector: {e}"
                self._estado.corriendo = False
                self._estado.conectado = False
                self._estado.arrancando = False
            return

        fps, t_prev = 0.0, time.perf_counter()
        t_inicio, n_frame, t_ultimo_llm = time.perf_counter(), 0, 0.0
        rostros: list = []

        # Freno para archivos. Sin esto el motor consume el video a la maxima
        # velocidad que da el CPU (medido: 327 FPS) y el usuario ve el clip
        # entero en un parpadeo, ademas de quemar un nucleo sin motivo.
        # Camaras USB y RTSP ya vienen limitadas por el hardware.
        periodo_objetivo = 0.0
        if cfg.tipo == "archivo":
            fps_video = fuente.fps_nativo() or 25.0
            periodo_objetivo = 1.0 / max(fps_video, 1.0)

        try:
            while not self._parar.is_set():
                ok, frame = fuente.leer()
                if not ok:
                    with self._lock:
                        self._estado.conectado = False
                    if cfg.tipo == "archivo":
                        with self._lock:
                            self._estado.termino_solo = True
                        break
                    continue

                if cfg.espejo and cfg.tipo == "usb":
                    import cv2
                    frame = cv2.flip(frame, 1)

                n_frame += 1
                transcurrido = time.perf_counter() - t_inicio

                if n_frame % cfg.saltar == 0:
                    detector.umbral = cfg.umbral
                    rostros = detector.procesar(frame, int(transcurrido * 1000))

                    diag = (frame.shape[0] ** 2 + frame.shape[1] ** 2) ** 0.5
                    pistas = seguidor.actualizar(
                        [(r.bbox, r.calidad.yaw, r.calidad.pitch, r.emocion.dominante)
                         for r in rostros],
                        transcurrido,
                    )
                    for r, (pid, pista) in zip(rostros, sorted(pistas.items())):
                        r.pista_id = pid
                        r.comportamiento = pista.comportamiento(diag)

                    self.agregador.actualizar(transcurrido, rostros, pistas)

                    fiables = [r for r in rostros if r.fiable]
                    if self.interprete and fiables:
                        self.buffer.agregar(transcurrido, fiables[0])
                        if (cfg.llm_auto > 0
                                and transcurrido - t_ultimo_llm >= cfg.llm_auto
                                and not self.interprete.ocupado
                                and self.interprete.solicitar(fiables[0], self.buffer)):
                            t_ultimo_llm = transcurrido

                for r in rostros:
                    dibujar_rostro_vigilancia(
                        frame, r, cfg.mostrar_malla, cfg.mostrar_calidad
                    )
                if cfg.mostrar_hud:
                    dibujar_hud_vigilancia(
                        frame, rostros, self.agregador, fuente, cfg.umbral, fps
                    )

                ahora = time.perf_counter()
                dt = ahora - t_prev
                t_prev = ahora
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                if periodo_objetivo:
                    sobra = periodo_objetivo - (time.perf_counter() - ahora)
                    if sobra > 0:
                        time.sleep(sobra)

                res = self.agregador.resumen_ventana()
                with self._lock:
                    self._frame = frame
                    self._estado.fps = fps
                    self._estado.aforo = len(rostros)
                    self._estado.fiables = sum(1 for r in rostros if r.fiable)
                    self._estado.calidad_media = res["calidad_media"]
                    self._estado.dominante = res["dominante"]
                    self._estado.descartadas_pct = self._pct_descartadas()
                    self._estado.conectado = fuente.estado.conectada
                    self._estado.reconexiones = fuente.estado.reconexiones
                    self._estado.transcurrido = transcurrido
                    self._estado.alertas = list(self.agregador.alertas)[-4:]

        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._estado.error = f"{type(e).__name__}: {e}"
        finally:
            detector.cerrar()
            fuente.cerrar()
            if self.interprete:
                self.interprete.cerrar()
            with self._lock:
                self._estado.corriendo = False
                self._estado.conectado = False
                self._estado.arrancando = False
