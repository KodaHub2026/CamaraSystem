"""
detector.py
===========
Detección de rostro con MediaPipe Face Landmarker y clasificación emocional.

MediaPipe entrega por cada rostro 478 puntos 3D y 52 "blendshapes" — coeficientes
de 0 a 1 que miden qué tan activa está cada expresión. Este módulo se encarga de
la captura y delega la interpretación emocional a `emociones.ClasificadorEmociones`.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from . import calidad as mod_calidad
from .emociones import ClasificadorEmociones, LecturaEmocional

URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class Rostro:
    """Un rostro detectado y su lectura emocional."""

    id: int
    bbox: tuple[int, int, int, int]     # (x, y, ancho, alto) en pixeles
    emocion: LecturaEmocional
    blendshapes: dict[str, float]
    landmarks: list
    calidad: "mod_calidad.Calidad | None" = None
    comportamiento: object = None       # se llena en main tras el seguimiento

    @property
    def confianza_efectiva(self) -> float:
        """Confianza cruda degradada por las condiciones de captura."""
        base = self.emocion.confianza
        return base * self.calidad.factor if self.calidad else base

    @property
    def fiable(self) -> bool:
        return self.calidad is None or self.calidad.suficiente

    @property
    def etiqueta(self) -> str:
        if not self.fiable:
            return "CALIDAD INSUF."
        return self.emocion.etiqueta

    @property
    def color(self) -> tuple[int, int, int]:
        if not self.fiable:
            return (110, 110, 110)
        return self.emocion.color


def asegurar_modelo(ruta: Path) -> Path:
    """Descarga el modelo de MediaPipe la primera vez que se ejecuta."""
    ruta = Path(ruta)
    if ruta.exists() and ruta.stat().st_size > 0:
        return ruta

    ruta.parent.mkdir(parents=True, exist_ok=True)
    print(f"[info] Descargando modelo (~3.7 MB) a {ruta} ...")
    urllib.request.urlretrieve(URL_MODELO, ruta)
    print("[info] Modelo listo.")
    return ruta


class DetectorEmociones:
    """Face Landmarker + clasificador emocional, listo para video en vivo."""

    def __init__(
        self,
        ruta_modelo: str | Path = "modelos/face_landmarker.task",
        umbral: float = 0.22,
        num_rostros: int = 1,
        ventana_suavizado: int = 8,
    ) -> None:
        self.clasificador = ClasificadorEmociones(
            umbral=umbral, ventana_suavizado=ventana_suavizado
        )

        ruta = asegurar_modelo(Path(ruta_modelo))

        opciones = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ruta)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_rostros,
            output_face_blendshapes=True,          # <- indispensable
            output_facial_transformation_matrixes=True,   # necesario para pose
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opciones)

    # ------------------------------------------------------------------ #

    @property
    def umbral(self) -> float:
        return self.clasificador.umbral

    @umbral.setter
    def umbral(self, valor: float) -> None:
        self.clasificador.umbral = valor

    def procesar(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Rostro]:
        """Analiza un frame BGR (formato nativo de OpenCV) y regresa los rostros."""
        alto, ancho = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        resultado = self._landmarker.detect_for_video(imagen_mp, timestamp_ms)

        if not resultado.face_landmarks:
            self.clasificador.reiniciar()   # nadie en cuadro
            return []

        matrices = getattr(resultado, "facial_transformation_matrixes", None) or []

        rostros: list[Rostro] = []
        for idx, landmarks in enumerate(resultado.face_landmarks):
            bs = (
                {c.category_name: c.score for c in resultado.face_blendshapes[idx]}
                if idx < len(resultado.face_blendshapes)
                else {}
            )
            bbox = self._bbox(landmarks, ancho, alto)
            matriz = matrices[idx] if idx < len(matrices) else None
            rostros.append(
                Rostro(
                    id=idx,
                    bbox=bbox,
                    emocion=self.clasificador.clasificar(bs, idx),
                    blendshapes=bs,
                    landmarks=landmarks,
                    calidad=mod_calidad.evaluar(frame_bgr, bbox, matriz),
                )
            )
        return rostros

    def cerrar(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "DetectorEmociones":
        return self

    def __exit__(self, *_exc) -> None:
        self.cerrar()

    # ------------------------------------------------------------------ #

    @staticmethod
    def _bbox(landmarks, ancho: int, alto: int) -> tuple[int, int, int, int]:
        xs = np.array([p.x for p in landmarks]) * ancho
        ys = np.array([p.y for p in landmarks]) * alto
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        mx, my = int((x2 - x1) * 0.08), int((y2 - y1) * 0.08)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(ancho, x2 + mx), min(alto, y2 + my)
        return x1, y1, x2 - x1, y2 - y1
