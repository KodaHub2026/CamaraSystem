"""
catalogo.py
===========
Descubrimiento y prueba de modelos en LM Studio.

EL PROBLEMA QUE RESUELVE
------------------------
`GET /v1/models` devolvio 29 modelos en el servidor de KodaHub. De esos, uno
solo respondia: el resto fallaba con "model not loaded" porque el Just-In-Time
loading estaba apagado en LM Studio.

O sea: **aparecer en la lista no significa estar cargado**. Un desplegable que
solo muestre `/v1/models` invita a elegir modelos que no van a funcionar, y el
usuario se entera hasta que el sistema falla en medio de una demo.

Por eso cada modelo se puede PROBAR desde la interfaz. La prueba mide tres
cosas que la lista no dice:

    1. Si responde de verdad (esta cargado en memoria)
    2. Cuanta latencia tiene por el tunel
    3. Si razona, y por lo tanto necesita presupuesto alto de tokens

Todas las pruebas corren en hilos aparte: la interfaz nunca se congela, aunque
el modelo tarde 80 segundos.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import requests

from .llm import ClienteLMStudio, _extraer_json

# Categorias por latencia medida, para semaforo en la interfaz
RAPIDO_S = 12.0
LENTO_S = 45.0


@dataclass
class ResultadoPrueba:
    modelo: str
    disponible: bool
    latencia: float = 0.0
    razona: bool = False
    json_valido: bool = False
    mensaje: str = ""

    @property
    def categoria(self) -> str:
        if not self.disponible:
            return "no_cargado"
        if self.latencia < RAPIDO_S:
            return "rapido"
        if self.latencia < LENTO_S:
            return "medio"
        return "lento"

    @property
    def color(self) -> str:
        return {
            "no_cargado": "#E05252",
            "rapido": "#8DC63F",
            "medio": "#F0B429",
            "lento": "#E07B39",
        }[self.categoria]

    @property
    def resumen(self) -> str:
        if not self.disponible:
            return self.mensaje or "no cargado en memoria"
        partes = [f"{self.latencia:.0f}s"]
        if self.razona:
            partes.append("razona")
        if not self.json_valido:
            partes.append("JSON irregular")
        return "  ".join(partes)


def _sugerir_vision(modelos: list[str]) -> list[str]:
    return [m for m in modelos if any(k in m.lower() for k in ("-vl", "vision", "llava", "4v"))]


class CatalogoModelos:
    """Consulta y prueba modelos sin bloquear la interfaz."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.modelos: list[str] = []
        self.resultados: dict[str, ResultadoPrueba] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #

    def listar_async(self, al_terminar) -> None:
        """Consulta /v1/models en un hilo. Llama a al_terminar(lista, error)."""

        def trabajo():
            try:
                cliente = ClienteLMStudio(base_url=self.base_url)
                modelos = cliente.listar_modelos()
                # Los que no son de embeddings primero; esos no sirven para chat.
                modelos = [m for m in modelos if "embed" not in m.lower()]
                with self._lock:
                    self.modelos = modelos
                al_terminar(modelos, None)
            except requests.exceptions.RequestException as e:
                al_terminar([], f"Sin conexion: {type(e).__name__}")
            except Exception as e:  # noqa: BLE001
                al_terminar([], str(e))

        threading.Thread(target=trabajo, daemon=True).start()

    def probar_async(self, modelo: str, al_terminar, timeout: int = 100) -> None:
        """
        Prueba un modelo de verdad. Llama a al_terminar(ResultadoPrueba).

        Timeout de 100 s a proposito: es el limite de Cloudflare. Si un modelo
        no responde en ese margen, tampoco va a servir en produccion.
        """

        def trabajo():
            cliente = ClienteLMStudio(base_url=self.base_url, modelo=modelo, timeout=timeout)
            t0 = time.perf_counter()
            try:
                respuesta = cliente.chat(
                    'Responde SOLO este JSON, sin nada mas: {"ok": true}',
                    "Confirma que estas activo.",
                )
                lat = time.perf_counter() - t0
                res = ResultadoPrueba(
                    modelo=modelo,
                    disponible=bool(respuesta.strip()),
                    latencia=lat,
                    razona=cliente.ultimo_razonamiento > 0,
                    json_valido=_extraer_json(respuesta) is not None,
                    mensaje="" if respuesta.strip() else "respuesta vacia",
                )
            except requests.exceptions.HTTPError as e:
                cuerpo = e.response.text.lower() if e.response is not None else ""
                if "load" in cuerpo:
                    msg = "no cargado en LM Studio"
                elif e.response is not None and e.response.status_code == 524:
                    msg = "timeout del tunel (524)"
                else:
                    msg = f"HTTP {e.response.status_code if e.response else '?'}"
                res = ResultadoPrueba(modelo, False, time.perf_counter() - t0, mensaje=msg)
            except requests.exceptions.Timeout:
                res = ResultadoPrueba(modelo, False, timeout, mensaje="excede 100s")
            except requests.exceptions.RequestException as e:
                res = ResultadoPrueba(modelo, False, mensaje=f"sin conexion ({type(e).__name__})")
            except Exception as e:  # noqa: BLE001
                res = ResultadoPrueba(modelo, False, mensaje=str(e)[:50])

            with self._lock:
                self.resultados[modelo] = res
            al_terminar(res)

        threading.Thread(target=trabajo, daemon=True).start()

    def resultado(self, modelo: str) -> ResultadoPrueba | None:
        with self._lock:
            return self.resultados.get(modelo)

    def modelos_vision(self) -> list[str]:
        with self._lock:
            return _sugerir_vision(self.modelos)
