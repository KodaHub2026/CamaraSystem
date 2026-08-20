"""
main.py
=======
Analisis emocional para camara de vigilancia.

Uso:
    python main.py --rtsp "rtsp://usuario:clave@10.0.0.50:554/stream1"
    python main.py --camara 0 --aforo 8
    python main.py --rtsp "..." --sin-ventana --salida metricas.csv
    python main.py --video grabacion.mp4 --aforo 6

Controles (con ventana):
    Q / ESC   salir          M  malla facial
    E         espectro       C  detalle de calidad
    + / -     umbral         G  captura
    I         interpretar con Qwen (requiere --llm)
    S         resumen de sesion con Qwen
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from src.agregado import Agregador
from src.detector import DetectorEmociones
from src.fuente import FuenteVideo
from src.llm import (
    MODELO_POR_DEFECTO, URL_POR_DEFECTO,
    BufferTemporal, ClienteLMStudio, InterpreteAsincrono,
)
from src.seguimiento import Seguidor
from src.ui import dibujar_hud_vigilancia, dibujar_panel_llm, dibujar_rostro_vigilancia

RAIZ = Path(__file__).parent


def parsear() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analisis emocional para camara de vigilancia.")
    f = p.add_argument_group("fuente de video")
    f.add_argument("--rtsp", type=str, default=None, help="URL RTSP de la camara IP")
    f.add_argument("--camara", type=int, default=None, help="Indice de camara USB")
    f.add_argument("--video", type=str, default=None, help="Archivo de video a analizar")
    f.add_argument("--ancho", type=int, default=1280)
    f.add_argument("--alto", type=int, default=720)

    d = p.add_argument_group("deteccion")
    d.add_argument("--aforo", type=int, default=6, help="Maximo de personas simultaneas")
    d.add_argument("--umbral", type=float, default=0.22)
    d.add_argument("--suavizado", type=int, default=10)
    d.add_argument("--saltar", type=int, default=1, help="Analizar 1 de cada N frames")

    o = p.add_argument_group("operacion")
    o.add_argument("--sin-ventana", action="store_true", help="Modo desatendido, sin GUI")
    o.add_argument("--salida", type=str, default=None, help="CSV de metricas agregadas")
    o.add_argument("--ventana-s", type=float, default=60.0)
    o.add_argument("--alerta-aforo", type=int, default=0, help="Alertar con N+ personas")
    o.add_argument("--alerta-seg", type=float, default=4.0, help="Segundos sostenidos para alertar")
    o.add_argument("--malla", action="store_true")
    o.add_argument("--sin-espejo", action="store_true")

    l = p.add_argument_group("interpretacion LLM")
    l.add_argument("--llm", action="store_true")
    l.add_argument("--llm-url", default=URL_POR_DEFECTO)
    l.add_argument("--llm-modelo", default=MODELO_POR_DEFECTO)
    l.add_argument("--llm-auto", type=float, default=0.0)
    return p.parse_args()


def resolver_origen(args):
    if args.rtsp:
        return args.rtsp, "rtsp"
    if args.video:
        return args.video, "archivo"
    return (args.camara if args.camara is not None else 0), "usb"


def main() -> None:
    args = parsear()
    origen, tipo = resolver_origen(args)
    espejo = not args.sin_espejo and tipo == "usb"   # solo tiene sentido en webcam

    print(f"[info] Fuente: {tipo} -> {origen}")
    print(f"[info] Aforo maximo: {args.aforo} personas")

    agregador = Agregador(
        ventana_s=args.ventana_s, salida=args.salida,
        alerta_segundos=args.alerta_seg, alerta_aforo=args.alerta_aforo,
    )
    seguidor = Seguidor()

    interprete, buffer = None, BufferTemporal(ventana_seg=30.0)
    if args.llm:
        interprete = InterpreteAsincrono(
            ClienteLMStudio(base_url=args.llm_url, modelo=args.llm_modelo)
        )
        print(f"[info] LLM activo: {args.llm_modelo}")

    ventana = "Vigilancia Emocional - KodaHub"
    if not args.sin_ventana:
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)

    mostrar_malla, mostrar_calidad = args.malla, False
    umbral, fps = args.umbral, 0.0
    t_prev = t_inicio = time.perf_counter()
    n_frame, t_ultimo_llm = 0, 0.0
    rostros: list = []

    with FuenteVideo(origen, args.ancho, args.alto) as fuente, \
         DetectorEmociones(
             ruta_modelo=RAIZ / "modelos" / "face_landmarker.task",
             umbral=umbral, num_rostros=args.aforo,
             ventana_suavizado=args.suavizado,
         ) as detector:

        if not fuente.estado.conectada:
            raise SystemExit(f"[error] No se pudo abrir la fuente: {origen}")
        print("[info] Corriendo. Ctrl+C para detener.\n")

        try:
            while True:
                ok, frame = fuente.leer()
                if not ok:
                    if tipo == "archivo":
                        break
                    continue

                if espejo:
                    frame = cv2.flip(frame, 1)

                n_frame += 1
                transcurrido = time.perf_counter() - t_inicio

                # Submuestreo: en vigilancia rara vez hace falta analizar los 30
                # frames por segundo. Con --saltar 2 se duplica el aforo posible.
                if n_frame % args.saltar == 0:
                    detector.umbral = umbral
                    rostros = detector.procesar(frame, int(transcurrido * 1000))

                    diag = (frame.shape[0] ** 2 + frame.shape[1] ** 2) ** 0.5
                    pistas = seguidor.actualizar(
                        [(r.bbox, r.calidad.yaw, r.calidad.pitch, r.emocion.dominante)
                         for r in rostros],
                        transcurrido,
                    )
                    for r, (pid, pista) in zip(rostros, sorted(pistas.items())):
                        r.pista_id = pid
                        r.comportamiento = pista.comportamiento(diag)

                    agregador.actualizar(transcurrido, rostros, pistas)

                    if interprete and rostros:
                        fiables = [r for r in rostros if r.fiable]
                        if fiables:
                            buffer.agregar(transcurrido, fiables[0])
                            if (args.llm_auto > 0
                                    and transcurrido - t_ultimo_llm >= args.llm_auto
                                    and not interprete.ocupado
                                    and interprete.solicitar(fiables[0], buffer)):
                                t_ultimo_llm = transcurrido

                ahora = time.perf_counter()
                dt = ahora - t_prev
                t_prev = ahora
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                if args.sin_ventana:
                    if n_frame % 300 == 0:
                        r = agregador.resumen_ventana()
                        print(f"[{transcurrido:6.0f}s] aforo {r['aforo_medio']:.1f} | "
                              f"dominante {r['dominante']} | calidad {r['calidad_media']:.2f} | "
                              f"{fps:.1f} FPS", flush=True)
                    continue

                for r in rostros:
                    dibujar_rostro_vigilancia(frame, r, mostrar_malla, mostrar_calidad)
                dibujar_hud_vigilancia(frame, rostros, agregador, fuente, umbral, fps)
                if interprete:
                    dibujar_panel_llm(frame, interprete)

                cv2.imshow(ventana, frame)
                t = cv2.waitKey(1) & 0xFF
                if t in (ord("q"), ord("Q"), 27):
                    break
                elif t in (ord("m"), ord("M")):
                    mostrar_malla = not mostrar_malla
                elif t in (ord("c"), ord("C")):
                    mostrar_calidad = not mostrar_calidad
                elif t in (ord("+"), ord("=")):
                    umbral = min(0.90, round(umbral + 0.02, 2))
                elif t in (ord("-"), ord("_")):
                    umbral = max(0.04, round(umbral - 0.02, 2))
                elif t in (ord("g"), ord("G")):
                    c = RAIZ / "capturas"; c.mkdir(exist_ok=True)
                    n = c / f"captura_{datetime.now():%Y%m%d_%H%M%S}.png"
                    cv2.imwrite(str(n), frame); print(f"[info] {n}")
                elif t in (ord("i"), ord("I")) and interprete and rostros:
                    fiables = [r for r in rostros if r.fiable]
                    if fiables and interprete.solicitar(fiables[0], buffer):
                        print("[info] Consultando a Qwen (el video no se detiene)...")
                elif t in (ord("s"), ord("S")) and interprete:
                    interprete.solicitar_sesion(buffer, transcurrido)

        except KeyboardInterrupt:
            print("\n[info] Interrumpido por el usuario.")

    agregador.cerrar(time.perf_counter() - t_inicio)
    if interprete:
        interprete.cerrar()
    if not args.sin_ventana:
        cv2.destroyAllWindows()

    e = fuente.estado
    print(f"[fuente] {e.frames_leidos} frames leidos, "
          f"{e.frames_perdidos} perdidos, {e.reconexiones} reconexiones")


if __name__ == "__main__":
    main()
