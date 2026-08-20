"""
verificar_lm.py
===============
Diagnostico del tunel de LM Studio antes de correr el detector.

Ejecuta:
    python verificar_lm.py
    python verificar_lm.py --modelo google/gemma-4-12b
    python verificar_lm.py --url http://localhost:1234

Comprueba cuatro cosas:
  1. Que el tunel responda y que modelos hay
  2. Que el modelo elegido este REALMENTE cargado (listado != cargado)
  3. Latencia real de una llamada tipo
  4. Si el modelo razona y se come el presupuesto de tokens
"""

from __future__ import annotations

import argparse
import sys
import time

import requests

from src.llm import (
    MODELO_POR_DEFECTO,
    URL_POR_DEFECTO,
    ClienteLMStudio,
    _extraer_json,
)

VERDE, ROJO, AMARILLO, GRIS, FIN = "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[0m"
OK, MAL, AVISO = f"{VERDE}[OK]{FIN}", f"{ROJO}[FALLA]{FIN}", f"{AMARILLO}[AVISO]{FIN}"


def titulo(t: str) -> None:
    print(f"\n{t}\n" + "-" * len(t))


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnostico del tunel de LM Studio")
    p.add_argument("--url", default=URL_POR_DEFECTO)
    p.add_argument("--modelo", default=MODELO_POR_DEFECTO)
    args = p.parse_args()

    cliente = ClienteLMStudio(base_url=args.url, modelo=args.modelo)
    print(f"\nTunel:  {args.url}\nModelo: {args.modelo}")

    # ---------------------------------------------------------------- 1
    titulo("1. Conectividad")
    try:
        t0 = time.perf_counter()
        modelos = cliente.listar_modelos()
        print(f"{OK} Tunel accesible ({time.perf_counter() - t0:.2f}s) - {len(modelos)} modelos listados")
    except requests.exceptions.RequestException as e:
        print(f"{MAL} No se pudo conectar: {e}")
        print(f"\n{GRIS}Revisa que LM Studio este corriendo y el tunel de Cloudflare activo.{FIN}")
        return 1

    vision = [m for m in modelos if any(k in m.lower() for k in ("-vl", "vision", "llava", "4v"))]
    if vision:
        print(f"{GRIS}     Con vision: {', '.join(vision)}{FIN}")

    # ---------------------------------------------------------------- 2
    titulo("2. El modelo esta cargado")
    if args.modelo not in modelos:
        print(f"{MAL} '{args.modelo}' no aparece en la lista.")
        print(f"{GRIS}     Disponibles: {', '.join(modelos[:6])}...{FIN}")
        return 1
    print(f"{OK} Aparece en la lista")
    print(f"{GRIS}     Ojo: aparecer != estar cargado. Lo confirma la prueba 3.{FIN}")

    # ---------------------------------------------------------------- 3
    titulo("3. Latencia real")
    try:
        t0 = time.perf_counter()
        respuesta = cliente.chat(
            "Responde SOLO este JSON: {\"ok\": true}",
            "Confirma que estas activo.",
        )
        lat = time.perf_counter() - t0
    except requests.exceptions.HTTPError as e:
        cuerpo = e.response.text[:200] if e.response is not None else ""
        if "load" in cuerpo.lower() or "not loaded" in cuerpo.lower():
            print(f"{MAL} El modelo esta listado pero NO cargado en memoria.")
            print(f"{GRIS}     En LM Studio: pestana Developer -> carga el modelo,{FIN}")
            print(f"{GRIS}     o activa Just-In-Time model loading en Settings.{FIN}")
        else:
            print(f"{MAL} HTTP {e.response.status_code if e.response else '?'}: {cuerpo}")
        return 1
    except requests.exceptions.RequestException as e:
        print(f"{MAL} {type(e).__name__}: {e}")
        return 1

    marca = OK if lat < 15 else (AVISO if lat < 60 else MAL)
    print(f"{marca} Respondio en {lat:.1f}s  (streaming activo)")
    print(f"{GRIS}     El streaming es obligatorio con Cloudflare: sin el,{FIN}")
    print(f"{GRIS}     cualquier request de mas de ~100s recibe HTTP 524.{FIN}")

    if lat > 60:
        print(f"{AVISO} Muy lento para uso frecuente.")
        print(f"{GRIS}     Sirve bajo demanda (tecla I) y resumen de sesion (tecla S).{FIN}")
        print(f"{GRIS}     Para algo agil, carga un modelo chico en LM Studio.{FIN}")
    elif lat > 15:
        print(f"{GRIS}     Aceptable bajo demanda, no para uso continuo.{FIN}")

    # ---------------------------------------------------------------- 4
    titulo("4. Formato de respuesta")
    if not respuesta.strip():
        print(f"{MAL} Contenido VACIO.")
        print(f"{GRIS}     Tipico de modelos de razonamiento: gastan todo el{FIN}")
        print(f"{GRIS}     presupuesto en reasoning_tokens. Sube MAX_TOKENS en src/llm.py.{FIN}")
        return 1

    razona = "<think>" in respuesta.lower()
    if razona:
        print(f"{AVISO} Emite bloques <think> (modelo de razonamiento)")
        print(f"{GRIS}     Ya contemplado: _extraer_json() los descarta.{FIN}")

    if _extraer_json(respuesta) is not None:
        print(f"{OK} JSON parseado correctamente")
    else:
        print(f"{AVISO} No devolvio JSON valido")
        print(f"{GRIS}     Respuesta: {respuesta.strip()[:110]}{FIN}")
        print(f"{GRIS}     El sistema lo tolera: muestra el texto crudo.{FIN}")

    # ---------------------------------------------------------------- #
    titulo("Veredicto")
    if lat < 60:
        print(f"{VERDE}Listo para usar.{FIN}  python main.py --llm")
    else:
        print(f"{AMARILLO}Usable solo bajo demanda ({lat:.0f}s por consulta).{FIN}")
        print(f"El video sigue a 30 FPS; el LLM responde en segundo plano.")
        print(f"\n  python main.py --llm        luego tecla I (instante) o S (sesion)")
        if vision:
            print(f"\n{GRIS}Si necesitas algo mas rapido, prueba un modelo chico del{FIN}")
            print(f"{GRIS}mismo servidor:  python verificar_lm.py --modelo qwen/qwen3-vl-8b{FIN}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
