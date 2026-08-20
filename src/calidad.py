"""
calidad.py
==========
Portero de calidad. Decide si las condiciones permiten una lectura emocional
confiable, y degrada o bloquea el resultado cuando no.

POR QUE ESTE MODULO ES EL MAS IMPORTANTE DEL SISTEMA
----------------------------------------------------
MediaPipe estima blendshapes bien con un rostro grande, frontal e iluminado.
Una camara de seguridad entrega justo lo contrario:

    Webcam            Camara de vigilancia
    ---------------   -----------------------------
    rostro 350 px     rostro 35-80 px
    frontal           montada en alto, mira hacia abajo
    luz de pantalla   contraluz, IR nocturno
    1 persona         varias, a distintas distancias

El problema no es que la precision baje. Es que MediaPipe SIGUE devolviendo
numeros con la misma pinta de confianza. Un rostro de 30 px produce blendshapes
que son esencialmente ruido, y sin este modulo el sistema reportaria
"ENOJO 0.58" con total aplomo.

Peor: el pitch negativo (camara mirando hacia abajo) comprime verticalmente la
zona de las cejas, lo que el modelo lee como browDown -> ENOJO FALSO SISTEMATICO.
Es el sesgo mas peligroso de todo el pipeline, porque es consistente, no aleatorio.

Este modulo mide cuatro cosas y produce un FACTOR 0-1 que multiplica la
confianza. Por debajo de `UMBRAL_BLOQUEO` el sistema reporta "CALIDAD
INSUFICIENTE" en lugar de inventar una emocion.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Umbrales calibrados para MediaPipe Face Landmarker
ANCHO_MINIMO_PX = 42        # por debajo, los blendshapes son ruido
ANCHO_BUENO_PX = 110        # a partir de aqui, resolucion suficiente
YAW_MAXIMO = 32.0           # grados de giro horizontal tolerables
PITCH_MAXIMO = 24.0         # grados de cabeceo; mas alla distorsiona las cejas
NITIDEZ_MINIMA = 18.0       # varianza del Laplaciano
LUM_MINIMA, LUM_MAXIMA = 35.0, 225.0

UMBRAL_BLOQUEO = 0.35       # por debajo: no se reporta emocion


@dataclass
class Calidad:
    """Diagnostico de las condiciones de captura de un rostro."""

    ancho_px: int
    yaw: float                  # giro horizontal (grados)
    pitch: float                # cabeceo (grados); negativo = mira hacia abajo
    roll: float                 # inclinacion lateral (grados)
    nitidez: float
    luminancia: float
    factor: float               # 0-1, multiplica la confianza emocional
    motivos: list[str]          # que degrado la calidad

    @property
    def suficiente(self) -> bool:
        return self.factor >= UMBRAL_BLOQUEO

    @property
    def resumen(self) -> str:
        if self.suficiente and not self.motivos:
            return "OK"
        return ", ".join(self.motivos) if self.motivos else "OK"


def angulos_cabeza(matriz) -> tuple[float, float, float]:
    """
    Extrae yaw / pitch / roll de la matriz de transformacion facial 4x4
    que entrega MediaPipe. Descomposicion Euler XYZ estandar.
    """
    if matriz is None:
        return 0.0, 0.0, 0.0

    M = np.array(matriz).reshape(4, 4)
    R = M[:3, :3]

    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0

    return tuple(float(v) for v in np.degrees([pitch, yaw, roll]))


def _rampa(valor: float, malo: float, bueno: float) -> float:
    """Interpolacion lineal 0-1 entre un umbral malo y uno bueno."""
    if bueno == malo:
        return 1.0
    return float(np.clip((valor - malo) / (bueno - malo), 0.0, 1.0))


def evaluar(frame_bgr: np.ndarray, bbox, matriz_transformacion) -> Calidad:
    """Evalua las condiciones de captura de un rostro y produce el factor."""
    x, y, w, h = bbox
    motivos: list[str] = []

    pitch, yaw, roll = angulos_cabeza(matriz_transformacion)

    # --- 1. Resolucion --------------------------------------------------
    f_tam = _rampa(w, ANCHO_MINIMO_PX, ANCHO_BUENO_PX)
    if w < ANCHO_MINIMO_PX:
        motivos.append(f"rostro muy chico ({w}px)")
    elif w < ANCHO_BUENO_PX:
        motivos.append(f"resolucion baja ({w}px)")

    # --- 2. Angulo ------------------------------------------------------
    # Rampas mas agresivas que las de tamano o luz, a proposito. Un angulo
    # extremo no degrada la lectura: la INVIERTE. Un pitch de -40 grados
    # comprime la zona de las cejas y MediaPipe lo lee como browDown, o sea
    # ENOJO, en una persona perfectamente neutral. Ese sesgo es sistematico,
    # no aleatorio, asi que promediar frames no lo corrige: lo consolida.
    f_yaw = 1.0 - _rampa(abs(yaw), YAW_MAXIMO, YAW_MAXIMO * 1.6)
    f_pitch = 1.0 - _rampa(abs(pitch), PITCH_MAXIMO, PITCH_MAXIMO * 1.6)
    if abs(yaw) > YAW_MAXIMO:
        motivos.append(f"perfil {abs(yaw):.0f}deg")
    if abs(pitch) > PITCH_MAXIMO:
        # Caso critico de camara montada en alto.
        motivos.append(f"cabeceo {pitch:+.0f}deg")

    # Veto absoluto: mas alla de estos angulos no hay lectura que rescatar.
    if abs(yaw) > YAW_MAXIMO * 1.7 or abs(pitch) > PITCH_MAXIMO * 1.7:
        motivos.append("angulo inviable")
        return Calidad(w, yaw, pitch, roll, 0.0, 0.0, 0.0, motivos)

    # --- 3. Nitidez y luz ----------------------------------------------
    recorte = frame_bgr[max(0, y) : y + h, max(0, x) : x + w]
    if recorte.size == 0:
        return Calidad(w, yaw, pitch, roll, 0.0, 0.0, 0.0, ["fuera de cuadro"])

    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    luminancia = float(gris.mean())

    f_nitidez = _rampa(nitidez, NITIDEZ_MINIMA * 0.4, NITIDEZ_MINIMA * 3.0)
    if nitidez < NITIDEZ_MINIMA:
        motivos.append("movido o desenfocado")

    if luminancia < LUM_MINIMA:
        f_luz = _rampa(luminancia, 8.0, LUM_MINIMA)
        motivos.append("poca luz")
    elif luminancia > LUM_MAXIMA:
        f_luz = 1.0 - _rampa(luminancia, LUM_MAXIMA, 252.0)
        motivos.append("sobreexpuesto")
    else:
        f_luz = 1.0

    # --- Factor combinado ----------------------------------------------
    # Multiplicativo a proposito: un solo factor malo debe hundir el resultado.
    # Si el rostro mide 30 px, no importa que la luz sea perfecta.
    factor = float(f_tam * f_yaw * f_pitch * f_nitidez * f_luz)

    return Calidad(
        ancho_px=w, yaw=yaw, pitch=pitch, roll=roll,
        nitidez=nitidez, luminancia=luminancia,
        factor=factor, motivos=motivos,
    )
