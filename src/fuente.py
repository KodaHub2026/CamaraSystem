"""
fuente.py
=========
Fuente de video con reconexion automatica.

Una camara de seguridad no es una webcam: el stream RTSP se cae. Corte de red,
reinicio del NVR, saturacion del switch. Un `cv2.VideoCapture` crudo simplemente
empieza a devolver `False` para siempre y el programa se queda mudo.

Este modulo reconecta con retroceso exponencial y lleva la cuenta, para que un
sistema desatendido sobreviva la noche.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import cv2


@dataclass
class EstadoFuente:
    conectada: bool = False
    reconexiones: int = 0
    frames_leidos: int = 0
    frames_perdidos: int = 0
    ultimo_error: str = ""


class FuenteVideo:
    """
    Envuelve VideoCapture con reconexion. Acepta:
        - indice de webcam:  FuenteVideo(0)
        - RTSP:              FuenteVideo("rtsp://user:pass@10.0.0.50:554/stream1")
        - archivo de video:  FuenteVideo("grabacion.mp4")
    """

    def __init__(
        self,
        origen,
        ancho: int = 1280,
        alto: int = 720,
        reintento_max_s: float = 30.0,
        latencia_baja: bool = True,
        backend: int | None = None,
    ) -> None:
        self.backend = backend
        self.origen = origen
        self.ancho = ancho
        self.alto = alto
        self.reintento_max_s = reintento_max_s
        self.latencia_baja = latencia_baja
        self.estado = EstadoFuente()
        self._cap: cv2.VideoCapture | None = None
        self._espera = 1.0
        self._es_rtsp = isinstance(origen, str) and origen.lower().startswith("rtsp")

    # ------------------------------------------------------------------ #

    def abrir(self) -> bool:
        self.cerrar()

        if isinstance(self.origen, int):
            # Si la deteccion ya nos dijo que backend funciona, lo usamos.
            # Si no, los probamos en orden: una misma camara puede abrir con
            # DirectShow y fallar con Media Foundation segun el driver.
            from .camaras import BACKENDS
            candidatos = ([self.backend] if self.backend is not None
                          else [b for b, _ in BACKENDS])
            cap = None
            for b in candidatos:
                intento = cv2.VideoCapture(self.origen, b)
                if intento.isOpened():
                    ok, _f = intento.read()
                    if ok:
                        cap = intento
                        self.backend = b
                        break
                intento.release()
            if cap is None:
                cap = cv2.VideoCapture(self.origen, cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)
        else:
            cap = cv2.VideoCapture(self.origen, cv2.CAP_FFMPEG)

        if self.latencia_baja:
            # Buffer de 1: preferimos saltarnos frames antes que analizar
            # imagenes viejas. En vigilancia importa el ahora, no la secuencia.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.estado.conectada = False
            self.estado.ultimo_error = (
                "no se pudo abrir (probados: "
                + ", ".join(n for _b, n in __import__(
                    "src.camaras", fromlist=["BACKENDS"]).BACKENDS) + ")"
                if isinstance(self.origen, int) else "no se pudo abrir"
            )
            return False

        self._cap = cap
        self.estado.conectada = True
        self._espera = 1.0
        return True

    def leer(self):
        """Devuelve (ok, frame). Reconecta sola si el stream se cae."""
        if self._cap is None or not self.estado.conectada:
            if not self._reconectar():
                return False, None

        ok, frame = self._cap.read()
        if ok and frame is not None:
            self.estado.frames_leidos += 1
            return True, frame

        self.estado.frames_perdidos += 1
        self.estado.conectada = False
        self.estado.ultimo_error = "stream interrumpido"
        return False, None

    def _reconectar(self) -> bool:
        if not self._es_rtsp and not isinstance(self.origen, int):
            return False  # archivo terminado: no tiene sentido reintentar

        print(f"[fuente] Reconectando en {self._espera:.0f}s...", flush=True)
        time.sleep(self._espera)
        # Retroceso exponencial con techo, para no martillar el NVR.
        self._espera = min(self._espera * 2, self.reintento_max_s)

        if self.abrir():
            self.estado.reconexiones += 1
            print(f"[fuente] Reconectada (intento {self.estado.reconexiones})", flush=True)
            return True
        return False

    def fps_nativo(self) -> float:
        """FPS declarados por la fuente. 0 si no los reporta."""
        if self._cap is None:
            return 0.0
        v = self._cap.get(cv2.CAP_PROP_FPS)
        return float(v) if v and 0 < v < 200 else 0.0

    def cerrar(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.estado.conectada = False

    def __enter__(self) -> "FuenteVideo":
        self.abrir()
        return self

    def __exit__(self, *_exc) -> None:
        self.cerrar()
