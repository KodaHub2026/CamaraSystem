"""
registro.py
===========
Registro de la sesión a CSV: una fila por frame con la distribución emocional.

Sirve para analizar después cómo evolucionó la emoción a lo largo del tiempo
(reacción a un video, entrevista, demo de producto). Se abre con pandas o Excel.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .emociones import ORDEN


class RegistroCSV:
    """Escribe una fila por rostro por frame. Se activa/desactiva en caliente."""

    def __init__(self, ruta: str | Path) -> None:
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._archivo = self.ruta.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._archivo)
        self._writer.writerow(
            ["timestamp", "segundos", "rostro", "dominante", "confianza", *ORDEN]
        )
        self._filas = 0

    def escribir(self, segundos: float, rostros) -> None:
        ahora = datetime.now().isoformat(timespec="milliseconds")
        for r in rostros:
            self._writer.writerow(
                [
                    ahora,
                    f"{segundos:.3f}",
                    r.id,
                    r.emocion.dominante,
                    f"{r.emocion.confianza:.4f}",
                    *[f"{r.emocion.puntajes[k]:.4f}" for k in ORDEN],
                ]
            )
            self._filas += 1

    def cerrar(self) -> None:
        self._archivo.close()
        print(f"[info] Registro guardado: {self.ruta}  ({self._filas} filas)")
