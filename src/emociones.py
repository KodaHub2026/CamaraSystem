"""
emociones.py
============
Motor de clasificación de emociones a partir de blendshapes faciales.

Fundamento: FACS / EMFACS
-------------------------
Paul Ekman y Wallace Friesen desarrollaron el Facial Action Coding System (FACS),
que descompone cualquier expresión facial en "Action Units" (AU) — movimientos
musculares individuales. EMFACS mapea combinaciones de AUs a las siete emociones
básicas universales.

Los 52 blendshapes de MediaPipe son, en la práctica, aproximaciones de esas AUs.
Este módulo hace el puente: blendshapes -> AUs -> emoción.

Mapeo EMFACS implementado
-------------------------
    Felicidad   AU6 + AU12                  mejillas elevadas + comisuras arriba
    Tristeza    AU1 + AU4 + AU15            ceja interna arriba + comisuras abajo
    Enojo       AU4 + AU5 + AU7 + AU23      cejas abajo + parpados tensos + labios apretados
    Sorpresa    AU1 + AU2 + AU5 + AU26      cejas arriba + ojos abiertos + mandibula caida
    Miedo       AU1+2+4 + AU5 + AU20 + AU26 sorpresa + labios estirados + cejas juntas
    Asco        AU9 + AU10                  nariz arrugada + labio superior elevado
    Desprecio   AU12 + AU14 unilateral      asimetria: UNA comisura arriba

Pares confusos
--------------
    Miedo vs Sorpresa -> ambos abren ojos y mandibula. Los diferencia AU20
        (labios estirados horizontalmente) y AU4 (cejas juntas), presentes solo
        en miedo. Por eso sorpresa penaliza esas dos señales.

    Enojo vs Asco -> ambos arrugan la nariz. Los diferencia AU4 (cejas abajo),
        dominante en enojo, contra AU9+AU10, dominantes en asco.

    Desprecio vs Felicidad -> la clave es la ASIMETRIA. Una sonrisa levanta
        ambas comisuras; el desprecio, solo una. Como MediaPipe entrega blendshapes
        izquierdo y derecho por separado, la diferencia es medible directamente.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Definición de emociones (colores en BGR para OpenCV)
# --------------------------------------------------------------------------- #

NEUTRAL = "neutral"

REGLAS: dict[str, dict] = {
    "felicidad": {
        "etiqueta": "FELICIDAD",
        "color": (90, 220, 120),      # verde
        "au": "AU6+AU12",
        "señales": [
            (["mouthSmileLeft", "mouthSmileRight"], 0.65),
            (["cheekSquintLeft", "cheekSquintRight"], 0.35),
        ],
        "penaliza": [
            (["mouthFrownLeft", "mouthFrownRight"], 0.60),
            (["_cejasAbajo"], 0.30),
        ],
    },
    "tristeza": {
        "etiqueta": "TRISTEZA",
        "color": (200, 130, 70),      # azul
        "au": "AU1+AU4+AU15",
        "señales": [
            (["mouthFrownLeft", "mouthFrownRight"], 0.45),
            (["browInnerUp"], 0.35),
            (["mouthShrugLower"], 0.20),
        ],
        "penaliza": [
            (["mouthSmileLeft", "mouthSmileRight"], 0.70),
            (["jawOpen"], 0.25),
            (["eyeWideLeft", "eyeWideRight"], 0.20),
        ],
    },
    "enojo": {
        "etiqueta": "ENOJO",
        "color": (60, 60, 240),       # rojo
        "au": "AU4+AU5+AU7+AU23",
        "señales": [
            (["_cejasAbajo"], 0.42),
            (["eyeSquintLeft", "eyeSquintRight"], 0.24),
            (["mouthPressLeft", "mouthPressRight"], 0.22),
            (["noseSneerLeft", "noseSneerRight"], 0.12),
        ],
        "penaliza": [
            (["mouthSmileLeft", "mouthSmileRight"], 0.70),
            (["browInnerUp"], 0.45),
            (["_labioSuperiorArriba"], 0.30),   # eso seria asco, no enojo
        ],
    },
    "sorpresa": {
        "etiqueta": "SORPRESA",
        "color": (60, 200, 250),      # ambar
        "au": "AU1+AU2+AU5+AU26",
        "señales": [
            (["browInnerUp"], 0.22),
            (["browOuterUpLeft", "browOuterUpRight"], 0.26),
            (["eyeWideLeft", "eyeWideRight"], 0.26),
            (["jawOpen"], 0.26),
        ],
        "penaliza": [
            (["_cejasAbajo"], 0.65),                                # eso seria miedo
            (["mouthStretchLeft", "mouthStretchRight"], 0.50),      # eso seria miedo
            (["mouthSmileLeft", "mouthSmileRight"], 0.30),
        ],
    },
    "miedo": {
        "etiqueta": "MIEDO",
        "color": (190, 90, 200),      # morado
        "au": "AU1+2+4+5+20+26",
        "señales": [
            (["eyeWideLeft", "eyeWideRight"], 0.30),
            (["mouthStretchLeft", "mouthStretchRight"], 0.30),
            (["browInnerUp"], 0.22),
            (["jawOpen"], 0.18),
        ],
        "penaliza": [
            (["mouthSmileLeft", "mouthSmileRight"], 0.60),
            (["eyeSquintLeft", "eyeSquintRight"], 0.25),
        ],
    },
    "asco": {
        "etiqueta": "ASCO",
        "color": (110, 180, 140),     # verde olivo
        "au": "AU9+AU10",
        "señales": [
            (["noseSneerLeft", "noseSneerRight"], 0.45),
            (["_labioSuperiorArriba"], 0.35),
            (["mouthShrugUpper"], 0.20),
        ],
        "penaliza": [
            (["mouthSmileLeft", "mouthSmileRight"], 0.45),
            (["jawOpen"], 0.20),
        ],
    },
    "desprecio": {
        "etiqueta": "DESPRECIO",
        "color": (140, 140, 220),     # rosa palo
        "au": "AU12+AU14 unilateral",
        "señales": [
            (["_asimetriaSonrisa"], 0.55),
            (["_asimetriaHoyuelo"], 0.25),
            (["mouthDimpleLeft", "mouthDimpleRight"], 0.20),
        ],
        "penaliza": [
            # Si AMBAS comisuras suben, es sonrisa — no desprecio.
            (["_sonrisaSimetrica"], 0.85),
        ],
    },
}

COLOR_NEUTRAL = (150, 150, 150)
ETIQUETA_NEUTRAL = "NEUTRAL"

ORDEN = list(REGLAS.keys())


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #


@dataclass
class LecturaEmocional:
    """Distribución emocional de un rostro en un instante."""

    dominante: str                      # clave de la emoción, o "neutral"
    confianza: float                    # 0-1, activación de la dominante
    puntajes: dict[str, float]          # activación por emoción (suavizada)
    distribucion: dict[str, float]      # normalizada, suma 1.0 (para gráficas)

    @property
    def etiqueta(self) -> str:
        if self.dominante == NEUTRAL:
            return ETIQUETA_NEUTRAL
        return REGLAS[self.dominante]["etiqueta"]

    @property
    def color(self) -> tuple[int, int, int]:
        if self.dominante == NEUTRAL:
            return COLOR_NEUTRAL
        return REGLAS[self.dominante]["color"]

    @property
    def au(self) -> str:
        if self.dominante == NEUTRAL:
            return "-"
        return REGLAS[self.dominante]["au"]

    def top(self, n: int = 3) -> list[tuple[str, float]]:
        """Las n emociones más activas, de mayor a menor."""
        return sorted(self.puntajes.items(), key=lambda kv: -kv[1])[:n]


# --------------------------------------------------------------------------- #
# Motor
# --------------------------------------------------------------------------- #


class ClasificadorEmociones:
    """Convierte blendshapes en una distribución de emociones, con suavizado."""

    def __init__(
        self,
        umbral: float = 0.22,
        ventana_suavizado: int = 8,
        margen_cambio: float = 0.06,
        frames_confirmacion: int = 3,
    ) -> None:
        """
        umbral              activación mínima para no reportar NEUTRAL
        ventana_suavizado   frames del promedio móvil
        margen_cambio       cuánto debe superar una emoción a la actual para destronarla
        frames_confirmacion cuántos frames seguidos debe mantener esa ventaja
        """
        self.umbral = umbral
        self.ventana_suavizado = ventana_suavizado
        self.margen_cambio = margen_cambio
        self.frames_confirmacion = frames_confirmacion

        self._historial: dict[int, dict[str, deque]] = {}
        self._dominante: dict[int, str] = {}
        self._candidato: dict[int, tuple[str, int]] = {}

    # ---------------------------------------------------------------- #

    def clasificar(self, blendshapes: dict[str, float], id_rostro: int = 0) -> LecturaEmocional:
        bs = self._derivadas(blendshapes)

        crudos = {
            nombre: self._activacion(bs, regla) for nombre, regla in REGLAS.items()
        }
        suaves = {n: self._suavizar(id_rostro, n, v) for n, v in crudos.items()}

        dominante, confianza = self._elegir_dominante(id_rostro, suaves)

        return LecturaEmocional(
            dominante=dominante,
            confianza=confianza,
            puntajes=suaves,
            distribucion=self._normalizar(suaves),
        )

    def reiniciar(self) -> None:
        """Limpia el estado cuando no hay nadie en cuadro."""
        self._historial.clear()
        self._dominante.clear()
        self._candidato.clear()

    # ---------------------------------------------------------------- #
    # Internos
    # ---------------------------------------------------------------- #

    @staticmethod
    def _derivadas(bs: dict[str, float]) -> dict[str, float]:
        """Agrega features calculadas al diccionario, como si fueran blendshapes."""
        b = dict(bs)
        g = b.get

        sl, sr = g("mouthSmileLeft", 0.0), g("mouthSmileRight", 0.0)
        dl, dr = g("mouthDimpleLeft", 0.0), g("mouthDimpleRight", 0.0)

        b["_cejasAbajo"] = (g("browDownLeft", 0.0) + g("browDownRight", 0.0)) / 2.0
        b["_labioSuperiorArriba"] = (
            g("mouthUpperUpLeft", 0.0) + g("mouthUpperUpRight", 0.0)
        ) / 2.0
        b["_asimetriaSonrisa"] = abs(sl - sr)
        b["_asimetriaHoyuelo"] = abs(dl - dr)
        b["_sonrisaSimetrica"] = min(sl, sr)   # alto solo si AMBAS comisuras suben
        return b

    @staticmethod
    def _promedio(bs: dict[str, float], nombres: list[str]) -> float:
        return float(np.mean([bs.get(n, 0.0) for n in nombres]))

    def _activacion(self, bs: dict[str, float], regla: dict) -> float:
        señales = regla["señales"]
        peso_total = sum(w for _, w in señales)
        valor = sum(self._promedio(bs, g) * w for g, w in señales) / peso_total

        for grupo, peso in regla.get("penaliza", []):
            valor -= self._promedio(bs, grupo) * peso

        return float(np.clip(valor, 0.0, 1.0))

    def _suavizar(self, id_rostro: int, emocion: str, valor: float) -> float:
        por_rostro = self._historial.setdefault(id_rostro, {})
        hist = por_rostro.setdefault(emocion, deque(maxlen=self.ventana_suavizado))
        hist.append(valor)
        return float(sum(hist) / len(hist))

    def _elegir_dominante(self, id_rostro: int, puntajes: dict[str, float]):
        """
        Histéresis: la emoción actual se mantiene salvo que otra la supere por
        `margen_cambio` durante `frames_confirmacion` frames seguidos. Sin esto,
        la etiqueta parpadea entre dos emociones cercanas.
        """
        mejor, mejor_valor = max(puntajes.items(), key=lambda kv: kv[1])

        if mejor_valor < self.umbral:
            self._dominante[id_rostro] = NEUTRAL
            self._candidato.pop(id_rostro, None)
            return NEUTRAL, mejor_valor

        actual = self._dominante.get(id_rostro, NEUTRAL)

        if actual == NEUTRAL or actual == mejor:
            self._dominante[id_rostro] = mejor
            self._candidato.pop(id_rostro, None)
            return mejor, mejor_valor

        ventaja = mejor_valor - puntajes.get(actual, 0.0)
        if ventaja < self.margen_cambio:
            self._candidato.pop(id_rostro, None)
            return actual, puntajes[actual]

        cand, racha = self._candidato.get(id_rostro, (mejor, 0))
        racha = racha + 1 if cand == mejor else 1
        self._candidato[id_rostro] = (mejor, racha)

        if racha >= self.frames_confirmacion:
            self._dominante[id_rostro] = mejor
            self._candidato.pop(id_rostro, None)
            return mejor, mejor_valor

        return actual, puntajes[actual]

    def _normalizar(self, puntajes: dict[str, float]) -> dict[str, float]:
        """Distribución que suma 1.0, incluyendo neutral como residuo."""
        d = dict(puntajes)
        d[NEUTRAL] = float(np.clip(1.0 - max(puntajes.values()) / max(self.umbral, 1e-6), 0.0, 1.0))
        total = sum(d.values())
        if total <= 1e-6:
            return {k: (1.0 if k == NEUTRAL else 0.0) for k in d}
        return {k: v / total for k, v in d.items()}
