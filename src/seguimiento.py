"""
seguimiento.py
==============
Seguimiento multipersona con IDs persistentes y lectura de comportamiento.

Una camara de seguridad ve varias personas a la vez, entrando y saliendo de
cuadro. MediaPipe no da identidad: entrega rostros en un orden que puede cambiar
entre frames. Sin seguimiento, la persona A puede recibir el historial emocional
de la persona B en el frame siguiente.

Este modulo asigna IDs estables por solapamiento de cajas (IoU) y, sobre esa
trayectoria, deriva senales de COMPORTAMIENTO que el rostro por si solo no da:

    permanencia   cuanto lleva la persona en cuadro
    agitacion     variabilidad del movimiento de cabeza
    inquietud     desplazamiento del centro de la caja
    acercamiento  si el rostro crece (se aproxima) o encoge (se aleja)
    cabeza_baja   pitch sostenido hacia abajo

Estas senales vienen de la trayectoria del rostro y del pose de cabeza, sin
modelos extra. Para postura corporal completa haria falta MediaPipe Pose
Landmarker, que en modo multipersona a distancia cuesta demasiado computo para
lo que aporta aqui.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


def _iou(a, b) -> float:
    """Interseccion sobre union de dos cajas (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Comportamiento:
    """Senales derivadas de la trayectoria, no del rostro."""

    permanencia_s: float = 0.0
    agitacion: float = 0.0        # 0-1, variabilidad de rotacion de cabeza
    inquietud: float = 0.0        # 0-1, desplazamiento del centro
    acercamiento: float = 0.0     # >0 se acerca, <0 se aleja
    cabeza_baja: bool = False

    @property
    def notas(self) -> list[str]:
        n = []
        if self.agitacion > 0.55:
            n.append("agitado")
        if self.inquietud > 0.55:
            n.append("inquieto")
        if self.acercamiento > 0.35:
            n.append("se acerca")
        elif self.acercamiento < -0.35:
            n.append("se aleja")
        if self.cabeza_baja:
            n.append("cabeza baja")
        if self.permanencia_s > 45:
            n.append(f"{self.permanencia_s:.0f}s en cuadro")
        return n


@dataclass
class Pista:
    """Una persona rastreada a lo largo del tiempo."""

    id: int
    bbox: tuple
    t_inicio: float
    t_visto: float
    frames_perdida: int = 0
    hist_centro: deque = field(default_factory=lambda: deque(maxlen=45))
    hist_area: deque = field(default_factory=lambda: deque(maxlen=45))
    hist_yaw: deque = field(default_factory=lambda: deque(maxlen=45))
    hist_pitch: deque = field(default_factory=lambda: deque(maxlen=45))
    hist_emocion: deque = field(default_factory=lambda: deque(maxlen=300))

    def actualizar(self, bbox, t, yaw, pitch, emocion) -> None:
        self.bbox = bbox
        self.t_visto = t
        self.frames_perdida = 0
        x, y, w, h = bbox
        self.hist_centro.append((x + w / 2, y + h / 2))
        self.hist_area.append(float(w * h))
        self.hist_yaw.append(yaw)
        self.hist_pitch.append(pitch)
        self.hist_emocion.append(emocion)

    def comportamiento(self, diagonal_frame: float) -> Comportamiento:
        c = Comportamiento(permanencia_s=self.t_visto - self.t_inicio)

        if len(self.hist_yaw) >= 8:
            # Agitacion: desviacion de la rotacion de cabeza, normalizada.
            var = float(np.std(self.hist_yaw) + np.std(self.hist_pitch))
            c.agitacion = float(np.clip(var / 28.0, 0.0, 1.0))

        if len(self.hist_centro) >= 8:
            pts = np.array(self.hist_centro)
            desp = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            c.inquietud = float(np.clip(desp / (diagonal_frame * 0.010), 0.0, 1.0))

        if len(self.hist_area) >= 12:
            # Tendencia del area: pendiente normalizada del ultimo tramo.
            a = np.array(self.hist_area)
            mitad = len(a) // 2
            ini, fin = float(a[:mitad].mean()), float(a[mitad:].mean())
            if ini > 0:
                c.acercamiento = float(np.clip((fin - ini) / ini, -1.0, 1.0))

        if len(self.hist_pitch) >= 10:
            c.cabeza_baja = float(np.mean(list(self.hist_pitch)[-10:])) < -20.0

        return c

    def emocion_predominante(self) -> tuple[str, float]:
        """Emocion mas frecuente en el historial y su proporcion."""
        if not self.hist_emocion:
            return "neutral", 0.0
        conteo: dict[str, int] = {}
        for e in self.hist_emocion:
            conteo[e] = conteo.get(e, 0) + 1
        clave = max(conteo, key=conteo.get)
        return clave, conteo[clave] / len(self.hist_emocion)


class Seguidor:
    """Asignador de IDs persistentes por IoU."""

    def __init__(self, iou_minimo: float = 0.25, max_perdida: int = 18) -> None:
        self.iou_minimo = iou_minimo
        self.max_perdida = max_perdida
        self.pistas: dict[int, Pista] = {}
        self._siguiente_id = 1

    def actualizar(self, detecciones: list[tuple], t: float) -> dict[int, Pista]:
        """
        detecciones: lista de (bbox, yaw, pitch, emocion)
        Devuelve el mapa de pistas activas.
        """
        libres = set(self.pistas.keys())
        asignadas: dict[int, int] = {}   # indice deteccion -> id pista

        # Emparejamiento voraz por mejor IoU. Suficiente para rostros, que no
        # se solapan tanto como cuerpos completos.
        pares = []
        for i, (bbox, *_rest) in enumerate(detecciones):
            for pid in libres:
                s = _iou(bbox, self.pistas[pid].bbox)
                if s >= self.iou_minimo:
                    pares.append((s, i, pid))
        pares.sort(reverse=True)

        usadas_det: set[int] = set()
        for _s, i, pid in pares:
            if i in usadas_det or pid not in libres:
                continue
            asignadas[i] = pid
            usadas_det.add(i)
            libres.discard(pid)

        # Detecciones sin pista: personas nuevas.
        for i, (bbox, yaw, pitch, emocion) in enumerate(detecciones):
            if i in asignadas:
                pid = asignadas[i]
            else:
                pid = self._siguiente_id
                self._siguiente_id += 1
                self.pistas[pid] = Pista(id=pid, bbox=bbox, t_inicio=t, t_visto=t)
            self.pistas[pid].actualizar(bbox, t, yaw, pitch, emocion)

        # Pistas no vistas: envejecen y se descartan.
        for pid in list(libres):
            self.pistas[pid].frames_perdida += 1
            if self.pistas[pid].frames_perdida > self.max_perdida:
                del self.pistas[pid]

        return {k: v for k, v in self.pistas.items() if v.frames_perdida == 0}

    def reiniciar(self) -> None:
        self.pistas.clear()
