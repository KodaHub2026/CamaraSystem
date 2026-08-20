"""
agregado.py
===========
Metricas agregadas y alertas para operacion desatendida.

DECISION DE DISENO: AGREGADO POR DEFECTO
----------------------------------------
Este modulo NO guarda rostros, ni embeddings, ni nada que permita reidentificar
a una persona entre sesiones. Los IDs del seguidor son efimeros: viven mientras
la persona esta en cuadro y mueren cuando sale. Reiniciar el programa reinicia
la numeracion.

Lo que persiste son conteos por ventana de tiempo: cuantas personas, cuanto
tiempo predomino cada emocion, cuantas lecturas se descartaron por calidad.

Esto no es una restriccion arbitraria. Un sistema que sigue individuos entre
sesiones es un sistema de identificacion biometrica, con obligaciones legales
distintas y mucho mas pesadas. Manteniendo el agregado, el sistema mide
"el ambiente de la sala" en lugar de "el estado de Fulano", que ademas es
para lo que la tecnologia realmente alcanza.
"""

from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .emociones import ORDEN


@dataclass
class Alerta:
    t: float
    tipo: str
    detalle: str
    hora: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class Agregador:
    """
    Acumula metricas por ventana y dispara alertas por condiciones sostenidas.

    Las alertas exigen persistencia (`frames_minimos`) a proposito: una lectura
    aislada de "enojo" en un frame no significa nada. Solo un patron sostenido
    en el tiempo merece atencion.
    """

    def __init__(
        self,
        ventana_s: float = 60.0,
        salida: str | Path | None = None,
        alerta_emociones: tuple = ("enojo", "miedo"),
        alerta_umbral: float = 0.45,
        alerta_segundos: float = 4.0,
        alerta_aforo: int = 0,
    ) -> None:
        self.ventana_s = ventana_s
        self.alerta_emociones = alerta_emociones
        self.alerta_umbral = alerta_umbral
        self.alerta_segundos = alerta_segundos
        self.alerta_aforo = alerta_aforo

        self.alertas: deque[Alerta] = deque(maxlen=60)
        self._muestras: deque = deque(maxlen=20000)
        self._sostenido: dict[int, tuple[str, float]] = {}
        self._alertado: set[tuple[int, str]] = set()

        self.total_lecturas = 0
        self.descartadas_calidad = 0
        self.pico_aforo = 0

        self._csv = None
        self._writer = None
        if salida:
            ruta = Path(salida)
            ruta.parent.mkdir(parents=True, exist_ok=True)
            self._csv = ruta.open("w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._csv)
            self._writer.writerow(
                ["timestamp", "segundos", "aforo", "dominante_sala",
                 "calidad_media", "descartadas", *ORDEN]
            )

    # ------------------------------------------------------------------ #

    def actualizar(self, t: float, rostros: list, pistas: dict) -> None:
        aforo = len(rostros)
        self.pico_aforo = max(self.pico_aforo, aforo)

        fiables = [r for r in rostros if r.fiable]
        self.total_lecturas += aforo
        self.descartadas_calidad += aforo - len(fiables)

        self._muestras.append(
            (t, aforo, [r.emocion.dominante for r in fiables],
             sum(r.calidad.factor for r in rostros) / aforo if aforo else 0.0)
        )

        self._revisar_alertas(t, rostros, pistas, aforo)

        if self._writer and fiables:
            self._escribir(t, aforo, fiables)

    def _revisar_alertas(self, t, rostros, pistas, aforo) -> None:
        if self.alerta_aforo and aforo >= self.alerta_aforo:
            self._emitir(t, "aforo", f"{aforo} personas en cuadro")

        for r in rostros:
            if not r.fiable:
                continue
            pid = getattr(r, "pista_id", r.id)
            dom = r.emocion.dominante

            if dom in self.alerta_emociones and r.confianza_efectiva >= self.alerta_umbral:
                clave, desde = self._sostenido.get(pid, (dom, t))
                if clave != dom:
                    clave, desde = dom, t
                self._sostenido[pid] = (clave, desde)

                if t - desde >= self.alerta_segundos and (pid, dom) not in self._alertado:
                    self._alertado.add((pid, dom))
                    self._emitir(
                        t, dom.upper(),
                        f"sostenido {t - desde:.0f}s (conf {r.confianza_efectiva:.2f})",
                    )
            else:
                self._sostenido.pop(pid, None)
                self._alertado.discard((pid, dom))

    def _emitir(self, t: float, tipo: str, detalle: str) -> None:
        a = Alerta(t=t, tipo=tipo, detalle=detalle)
        self.alertas.append(a)
        print(f"[ALERTA {a.hora}] {tipo}: {detalle}", flush=True)

    def _escribir(self, t, aforo, fiables) -> None:
        conteo = {k: 0.0 for k in ORDEN}
        for r in fiables:
            for k in ORDEN:
                conteo[k] += r.emocion.puntajes[k]
        n = len(fiables)
        dom = max(conteo, key=conteo.get) if n else "-"
        calidad_media = sum(r.calidad.factor for r in fiables) / n if n else 0.0

        self._writer.writerow([
            datetime.now().isoformat(timespec="seconds"), f"{t:.1f}", aforo, dom,
            f"{calidad_media:.3f}", self.descartadas_calidad,
            *[f"{conteo[k] / n:.4f}" for k in ORDEN],
        ])

    # ------------------------------------------------------------------ #

    def resumen_ventana(self) -> dict:
        """Estado de la ultima ventana, para el HUD y el resumen final."""
        if not self._muestras:
            return {"aforo_medio": 0.0, "dominante": "-", "calidad_media": 0.0}

        t_fin = self._muestras[-1][0]
        recientes = [m for m in self._muestras if t_fin - m[0] <= self.ventana_s]
        if not recientes:
            return {"aforo_medio": 0.0, "dominante": "-", "calidad_media": 0.0}

        conteo: dict[str, int] = {}
        for _t, _a, doms, _q in recientes:
            for d in doms:
                conteo[d] = conteo.get(d, 0) + 1

        return {
            "aforo_medio": sum(m[1] for m in recientes) / len(recientes),
            "dominante": max(conteo, key=conteo.get) if conteo else "-",
            "calidad_media": sum(m[3] for m in recientes) / len(recientes),
            "reparto": {k: v / sum(conteo.values()) for k, v in
                        sorted(conteo.items(), key=lambda kv: -kv[1])} if conteo else {},
        }

    def cerrar(self, duracion: float) -> None:
        if self._csv:
            self._csv.close()

        pct = (self.descartadas_calidad / self.total_lecturas * 100) if self.total_lecturas else 0.0
        r = self.resumen_ventana()
        print("\n" + "=" * 58)
        print("RESUMEN DE SESION")
        print("=" * 58)
        print(f"  Duracion             {duracion:.0f} s")
        print(f"  Lecturas totales     {self.total_lecturas}")
        print(f"  Descartadas calidad  {self.descartadas_calidad}  ({pct:.1f}%)")
        print(f"  Pico de aforo        {self.pico_aforo}")
        print(f"  Emocion predominante {r['dominante']}")
        print(f"  Calidad media        {r['calidad_media']:.2f}")
        if pct > 45:
            print("\n  AVISO: mas del 45% de las lecturas se descartaron.")
            print("  Revisa altura, angulo e iluminacion de la camara antes")
            print("  de dar peso a estos resultados.")
        print("=" * 58 + "\n")
