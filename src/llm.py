"""
llm.py
======
Capa de interpretación con LM Studio (Qwen) sobre el túnel de KodaHub.

POR QUÉ ES ASÍNCRONO
--------------------
Medición real contra lmmstudio5090.koda-cloud.org con qwen/qwen3.8-27b:

    Llamada trivial (10 tokens)      ->  6.1 s
    Interpretación realista          -> 36-45 s

El loop de video corre a 30 FPS = 33 ms por frame. Meter una llamada de 45 s
en ese loop congela la cámara casi un minuto. Por eso TODA consulta al LLM
ocurre en un hilo aparte: el video nunca se detiene y el resultado aparece
cuando está listo.

DOS TRAMPAS DEL MODELO (verificadas, no teóricas)
-------------------------------------------------
1. qwen3.8-27b es un modelo de RAZONAMIENTO. Con max_tokens=180 devolvió
   contenido VACÍO porque los 180 tokens se fueron en reasoning_tokens.
   Solución: max_tokens alto (>=600). Nunca lo bajes "para acelerar" — es
   justo lo que rompe la respuesta.

2. Emite bloques <think>...</think> antes del JSON. Hay que quitarlos antes
   de parsear. `_extraer_json()` se encarga.

Si el túnel está caído, el sistema sigue funcionando: MediaPipe y las reglas
FACS son 100% locales. El LLM es opcional por diseño.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import requests

from .emociones import ORDEN, REGLAS

URL_POR_DEFECTO = "https://lmmstudio5090.koda-cloud.org"
MODELO_POR_DEFECTO = "qwen/qwen3.8-27b"

# El modelo razona antes de responder. Presupuesto generoso o sale vacío.
MAX_TOKENS = 3000
TIMEOUT_S = 300


# --------------------------------------------------------------------------- #
# Cliente HTTP
# --------------------------------------------------------------------------- #


class ClienteLMStudio:
    """Cliente mínimo para la API compatible con OpenAI de LM Studio."""

    def __init__(
        self,
        base_url: str = URL_POR_DEFECTO,
        modelo: str = MODELO_POR_DEFECTO,
        timeout: int = TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.modelo = modelo
        self.timeout = timeout
        self.ultimo_razonamiento = 0
        self._sesion = requests.Session()

    def listar_modelos(self) -> list[str]:
        r = self._sesion.get(f"{self.base_url}/v1/models", timeout=20)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def chat(self, sistema: str, usuario: str, temperatura: float = 0.3) -> str:
        """
        Llamada en STREAMING. No es un capricho: es obligatorio con este tunel.

        Medido contra lmmstudio5090.koda-cloud.org:
            sin stream -> HTTP 524 a los 125 s  (Cloudflare corta el origen)
            con stream -> HTTP 200 a los 238 s  (los tokens mantienen viva la conexion)

        Cloudflare mata cualquier request cuyo origen no responda en ~100 s.
        Al hacer streaming ve datos fluyendo y nunca dispara el 524.

        Ademas cortamos apenas el JSON este completo: como el modelo razona
        primero y escribe despues, esperar a `finish_reason` desperdicia
        minutos generando texto que no vamos a usar.
        """
        payload = {
            "model": self.modelo,
            "messages": [
                {"role": "system", "content": sistema},
                {"role": "user", "content": usuario},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": temperatura,
            "stream": True,
            # No usamos response_format json_schema: LM Studio lo soporta de
            # forma inconsistente segun el modelo. Pedimos JSON por prompt y
            # parseamos defensivamente.
        }

        contenido: list[str] = []
        n_razonamiento = 0

        with self._sesion.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as r:
            r.raise_for_status()

            for linea in r.iter_lines(decode_unicode=True):
                if not linea or not linea.startswith("data: "):
                    continue
                cuerpo = linea[6:]
                if cuerpo.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(cuerpo)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

                # LM Studio separa el razonamiento del contenido real.
                if delta.get("reasoning_content"):
                    n_razonamiento += 1
                    continue

                trozo = delta.get("content")
                if not trozo:
                    continue
                contenido.append(trozo)

                # Corte anticipado: si ya cerro un JSON valido, no seguimos.
                parcial = "".join(contenido)
                if parcial.count("{") and parcial.count("{") == parcial.count("}"):
                    if _extraer_json(parcial) is not None:
                        break

        self.ultimo_razonamiento = n_razonamiento
        return "".join(contenido)


# --------------------------------------------------------------------------- #
# Parseo defensivo
# --------------------------------------------------------------------------- #

_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_RE_FENCE = re.compile(r"```(?:json)?|```")


def _extraer_json(texto: str) -> dict | None:
    """Quita bloques <think>, fences de markdown y rescata el primer objeto JSON."""
    if not texto:
        return None

    limpio = _RE_THINK.sub("", texto)
    limpio = _RE_FENCE.sub("", limpio).strip()

    inicio, fin = limpio.find("{"), limpio.rfind("}")
    if inicio == -1 or fin <= inicio:
        return None
    try:
        return json.loads(limpio[inicio : fin + 1])
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Buffer temporal
# --------------------------------------------------------------------------- #


@dataclass
class Muestra:
    t: float
    dominante: str
    puntajes: dict[str, float]


class BufferTemporal:
    """Guarda la evolución emocional reciente para dársela como contexto al LLM."""

    def __init__(self, ventana_seg: float = 25.0) -> None:
        self.ventana_seg = ventana_seg
        self._muestras: deque[Muestra] = deque(maxlen=2000)

    def agregar(self, t: float, rostro) -> None:
        self._muestras.append(Muestra(t, rostro.emocion.dominante, dict(rostro.emocion.puntajes)))

    def resumen(self, paso_seg: float = 2.0) -> str:
        """Línea de tiempo compacta: una etiqueta cada `paso_seg` segundos."""
        if not self._muestras:
            return "sin datos"

        t_fin = self._muestras[-1].t
        recientes = [m for m in self._muestras if t_fin - m.t <= self.ventana_seg]
        if not recientes:
            return "sin datos"

        bloques: list[str] = []
        t_bloque = recientes[0].t
        actual: list[str] = []
        for m in recientes:
            if m.t - t_bloque >= paso_seg and actual:
                bloques.append(f"{t_bloque - recientes[0].t:.0f}s:{max(set(actual), key=actual.count)}")
                t_bloque, actual = m.t, []
            actual.append(m.dominante)
        if actual:
            bloques.append(f"{t_bloque - recientes[0].t:.0f}s:{max(set(actual), key=actual.count)}")
        return " -> ".join(bloques)

    def estadisticas(self) -> dict[str, float]:
        """Proporción de tiempo en cada emoción."""
        if not self._muestras:
            return {}
        conteo: dict[str, int] = {}
        for m in self._muestras:
            conteo[m.dominante] = conteo.get(m.dominante, 0) + 1
        total = len(self._muestras)
        return {k: v / total for k, v in sorted(conteo.items(), key=lambda kv: -kv[1])}

    def limpiar(self) -> None:
        self._muestras.clear()

    def __len__(self) -> int:
        return len(self._muestras)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

SISTEMA_INSTANTE = """Eres un analista de expresion facial. Recibes mediciones \
objetivas del sistema FACS (Facial Action Coding System) y las interpretas.

Los blendshapes son coeficientes 0-1 que miden activacion muscular real, medidos \
por vision computacional. No inventes datos que no esten en la entrada.

Tu valor esta en los MATICES que las 7 categorias basicas no capturan: mezclas \
(sorpresa + incomodidad), emociones sociales (cortesia, escepticismo, resignacion), \
y el significado del patron temporal.

Responde UNICAMENTE con este JSON, sin texto antes ni despues:
{"matiz": "2-4 palabras", "lectura": "una frase de maximo 18 palabras", \
"confianza": "alta|media|baja"}"""

SISTEMA_SESION = """Eres un analista de expresion facial. Recibes el resumen de \
una sesion completa de medicion FACS y produces un informe breve.

Basate solo en los datos entregados. Si la evidencia es debil, dilo.

Responde UNICAMENTE con este JSON, sin texto antes ni despues:
{"resumen": "2-3 frases sobre el arco emocional", "momentos": ["hasta 3 momentos \
clave con su segundo"], "confianza": "alta|media|baja"}"""


def _prompt_instante(rostro, buffer: BufferTemporal) -> str:
    top_bs = sorted(rostro.blendshapes.items(), key=lambda kv: -kv[1])[:5]
    emociones = ", ".join(
        f"{REGLAS[k]['etiqueta'].lower()} {rostro.emocion.puntajes[k]:.2f}"
        for k in ORDEN
        if rostro.emocion.puntajes[k] > 0.03
    ) or "todas por debajo de 0.03"

    return (
        f"Dominante: {rostro.etiqueta} ({rostro.emocion.confianza:.2f})\n"
        f"Emociones: {emociones}\n"
        f"Musculos: " + ", ".join(f"{n} {v:.2f}" for n, v in top_bs)
        + f"\nTimeline: {buffer.resumen(paso_seg=4.0)}\n\nInterpreta el matiz."
    )


def _prompt_sesion(buffer: BufferTemporal, duracion: float) -> str:
    stats = buffer.estadisticas()
    reparto = ", ".join(f"{k} {v * 100:.0f}%" for k, v in list(stats.items())[:5])
    return (
        f"DURACION: {duracion:.0f} segundos, {len(buffer)} mediciones\n"
        f"REPARTO DE TIEMPO: {reparto}\n"
        f"LINEA DE TIEMPO: {buffer.resumen(paso_seg=3.0)}\n\n"
        "Produce el informe de la sesion."
    )


# --------------------------------------------------------------------------- #
# Intérprete asíncrono
# --------------------------------------------------------------------------- #


@dataclass
class Interpretacion:
    texto: str
    matiz: str = ""
    confianza: str = ""
    momentos: list = field(default_factory=list)
    segundos: float = 0.0
    error: bool = False
    hora: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class InterpreteAsincrono:
    """
    Consulta al LLM en un hilo aparte. El loop de video NUNCA se bloquea.

    Uso:
        interprete.solicitar(rostro, buffer)     # no bloquea, retorna al instante
        interprete.ultima                        # None hasta que llegue la respuesta
        interprete.ocupado                       # True mientras Qwen piensa
    """

    def __init__(self, cliente: ClienteLMStudio) -> None:
        self.cliente = cliente
        self._cola: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._ultima: Interpretacion | None = None
        self._ocupado = False
        self._activo = True

        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    # ---------------------------------------------------------------- #

    @property
    def ultima(self) -> Interpretacion | None:
        with self._lock:
            return self._ultima

    @property
    def ocupado(self) -> bool:
        with self._lock:
            return self._ocupado

    def solicitar(self, rostro, buffer: BufferTemporal) -> bool:
        """Encola una interpretación del instante. Si ya hay una en curso, la ignora."""
        return self._encolar(("instante", _prompt_instante(rostro, buffer)))

    def solicitar_desde_buffer(self, buffer: BufferTemporal) -> bool:
        """Interpreta usando solo la linea de tiempo, sin un rostro concreto."""
        if not len(buffer):
            return False
        stats = buffer.estadisticas()
        reparto = ", ".join(f"{k} {v * 100:.0f}%" for k, v in list(stats.items())[:4])
        prompt = (
            f"Reparto reciente: {reparto}\n"
            f"Timeline: {buffer.resumen(paso_seg=3.0)}\n\nInterpreta el matiz."
        )
        return self._encolar(("instante", prompt))

    def solicitar_sesion(self, buffer: BufferTemporal, duracion: float) -> bool:
        return self._encolar(("sesion", _prompt_sesion(buffer, duracion)))

    def cerrar(self) -> None:
        self._activo = False

    # ---------------------------------------------------------------- #

    def _encolar(self, item) -> bool:
        # OJO: `_ocupado` se marca AQUI, no en el hilo worker. Si se marcara alla,
        # entre encolar y que el hilo despierte habria una ventana donde
        # `ocupado` es False y el llamador cree que ya termino. Bug real,
        # detectado en pruebas: el consumidor salia del wait de inmediato.
        with self._lock:
            if self._ocupado:
                return False
            try:
                self._cola.put_nowait(item)
            except queue.Full:
                return False
            self._ocupado = True
            return True

    def _bucle(self) -> None:
        while self._activo:
            try:
                tipo, prompt = self._cola.get(timeout=0.4)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                sistema = SISTEMA_INSTANTE if tipo == "instante" else SISTEMA_SESION
                crudo = self.cliente.chat(sistema, prompt)
                datos = _extraer_json(crudo)
                resultado = self._formatear(tipo, datos, crudo, time.perf_counter() - t0)
            except requests.exceptions.Timeout:
                resultado = Interpretacion(
                    texto="Timeout: el modelo tardo demasiado", error=True,
                    segundos=time.perf_counter() - t0,
                )
            except requests.exceptions.RequestException as e:
                resultado = Interpretacion(
                    texto=f"Sin conexion al tunel ({type(e).__name__})", error=True,
                    segundos=time.perf_counter() - t0,
                )
            except Exception as e:  # noqa: BLE001
                resultado = Interpretacion(texto=f"Error: {e}", error=True)

            with self._lock:
                self._ultima = resultado
                self._ocupado = False

    @staticmethod
    def _formatear(tipo: str, datos: dict | None, crudo: str, segundos: float) -> Interpretacion:
        if datos is None:
            # El modelo respondio pero no en JSON valido: mostramos lo que dijo.
            limpio = _RE_THINK.sub("", crudo).strip()
            return Interpretacion(
                texto=limpio[:180] if limpio else "El modelo devolvio contenido vacio",
                confianza="baja", segundos=segundos, error=not limpio,
            )

        if tipo == "instante":
            return Interpretacion(
                texto=str(datos.get("lectura", "")),
                matiz=str(datos.get("matiz", "")),
                confianza=str(datos.get("confianza", "")),
                segundos=segundos,
            )

        return Interpretacion(
            texto=str(datos.get("resumen", "")),
            momentos=list(datos.get("momentos", []))[:3],
            confianza=str(datos.get("confianza", "")),
            segundos=segundos,
        )
