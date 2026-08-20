"""
camaras.py
==========
Deteccion automatica de camaras disponibles.

POR QUE NO BASTA UN CAMPO DE "INDICE"
-------------------------------------
Pedirle al usuario que adivine un numero es mal diseno: el indice depende del
orden en que Windows enumero los dispositivos, y cambia si conectas una webcam
USB, si hay una camara virtual de OBS instalada, o si el equipo trae una integrada.

Peor: en Windows el backend importa tanto como el indice. Una misma camara puede
abrir con DirectShow y fallar con Media Foundation, o al reves, segun el driver.
Probar solo un backend hace parecer que "no hay camara" cuando en realidad si la
hay, nomas por la puerta equivocada.

Este modulo prueba cada indice contra los tres backends y devuelve las que de
verdad entregan imagen, con su resolucion y el backend que funciono.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass

import cv2

# Orden de preferencia en Windows:
#   DSHOW  arranca rapido y es el mas compatible con webcams USB
#   MSMF   backend moderno; algunos drivers nuevos solo funcionan aqui
#   ANY    deja que OpenCV decida, ultimo recurso
if sys.platform == "win32":
    BACKENDS = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "MediaFoundation"),
        (cv2.CAP_ANY, "Auto"),
    ]
else:
    BACKENDS = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "Auto"),
    ]


@dataclass
class Camara:
    indice: int
    backend: int
    backend_nombre: str
    ancho: int
    alto: int

    @property
    def etiqueta(self) -> str:
        return f"Camara {self.indice}  ({self.ancho}x{self.alto}, {self.backend_nombre})"


def _probar(indice: int, backend: int) -> tuple[int, int] | None:
    """Abre e intenta LEER un frame. Abrir sin leer da falsos positivos."""
    cap = None
    # OpenCV escupe un WARN por cada indice y backend que no existe. Con 6
    # indices por 3 backends son 18 bloques de error en rojo para una busqueda
    # perfectamente normal. Los silenciamos: aqui el fallo es el caso esperado.
    nivel_previo = cv2.getLogLevel()
    cv2.setLogLevel(0)
    try:
        cap = cv2.VideoCapture(indice, backend)
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            return None
        alto, ancho = frame.shape[:2]
        return ancho, alto
    except Exception:  # noqa: BLE001
        return None
    finally:
        if cap is not None:
            cap.release()
        cv2.setLogLevel(nivel_previo)


def detectar(maximo: int = 6) -> list[Camara]:
    """Recorre indices 0..maximo-1 probando cada backend. Bloquea; usa detectar_async."""
    encontradas: list[Camara] = []
    for i in range(maximo):
        for backend, nombre in BACKENDS:
            res = _probar(i, backend)
            if res:
                encontradas.append(
                    Camara(indice=i, backend=backend, backend_nombre=nombre,
                           ancho=res[0], alto=res[1])
                )
                break   # ya funciono; no probamos los demas backends
    return encontradas


def detectar_async(al_terminar, maximo: int = 6) -> None:
    """
    Corre la deteccion en un hilo. Llama a al_terminar(lista).

    Es lento a proposito: abrir cada camara toma entre 200 ms y 2 s segun el
    driver. Con 6 indices y hasta 3 backends puede tardar varios segundos, y
    bloquear la interfaz durante ese rato la marcaria como "no responde".
    """
    def trabajo():
        try:
            al_terminar(detectar(maximo))
        except Exception:  # noqa: BLE001
            al_terminar([])

    threading.Thread(target=trabajo, daemon=True).start()
