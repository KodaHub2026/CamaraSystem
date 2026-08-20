"""
diagnostico.py
==============
Prueba cada capa del sistema por separado, en el hilo principal, mostrando el
error completo de la primera que falle.

    python diagnostico.py
    python diagnostico.py --camara 1

Por que existe: cuando el motor corre en un hilo de fondo, una excepcion queda
atrapada ahi y solo se ve un "FPS 0" sin explicacion. Este script hace lo mismo
paso a paso, sin hilos y sin interfaz, para que el error salga completo.
"""

from __future__ import annotations

import argparse
import platform
import sys
import traceback
from pathlib import Path

RAIZ = Path(__file__).parent
PASOS = 8


def paso(n: int, texto: str) -> None:
    print(f"\n[{n}/{PASOS}] {texto}")
    print("-" * 62)


def bien(msg: str) -> None:
    print(f"   OK   {msg}")


def mal(msg: str) -> None:
    print(f"   FALLA  {msg}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--camara", type=int, default=None)
    args = p.parse_args()

    print("=" * 62)
    print("DIAGNOSTICO — Vigilancia Emocional")
    print("=" * 62)

    # ---------------------------------------------------------------- 1
    paso(1, "Entorno")
    print(f"   Python      {sys.version.split()[0]}  ({platform.machine()})")
    print(f"   Sistema     {platform.system()} {platform.release()}")
    print(f"   Ejecutable  {sys.executable}")
    if sys.version_info >= (3, 13):
        print("\n   AVISO: Python 3.13+ es muy reciente. Si algo falla mas")
        print("   abajo, un entorno con 3.12 es la ruta probada:")
        print("       py -3.12 -m venv .venv")

    # ---------------------------------------------------------------- 2
    paso(2, "Importar dependencias")
    try:
        import cv2
        bien(f"opencv {cv2.__version__}")
        import numpy as np
        bien(f"numpy {np.__version__}")
        import mediapipe as mp
        bien(f"mediapipe {mp.__version__}")
    except Exception:
        mal("no se pudo importar")
        traceback.print_exc()
        return 1

    # ---------------------------------------------------------------- 3
    paso(3, "API de MediaPipe Tasks")
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        faltan = [
            n for n, ok in (
                ("BaseOptions", hasattr(mp_python, "BaseOptions")),
                ("FaceLandmarker", hasattr(vision, "FaceLandmarker")),
                ("FaceLandmarkerOptions", hasattr(vision, "FaceLandmarkerOptions")),
                ("RunningMode.VIDEO", hasattr(getattr(vision, "RunningMode", None), "VIDEO")),
                ("mp.Image", hasattr(mp, "Image")),
            ) if not ok
        ]
        if faltan:
            mal(f"API cambio; falta: {', '.join(faltan)}")
            print("\n   mediapipe 1.x movio el API. Ruta mas rapida:")
            print("       pip install \"mediapipe<1.0\"")
            return 1
        bien("todos los simbolos presentes")
    except Exception:
        mal("error importando tasks")
        traceback.print_exc()
        return 1

    # ---------------------------------------------------------------- 4
    paso(4, "Modelo face_landmarker.task")
    try:
        from src.detector import asegurar_modelo
        ruta = asegurar_modelo(RAIZ / "modelos" / "face_landmarker.task")
        bien(f"{ruta.stat().st_size / 1e6:.1f} MB en {ruta}")
    except Exception:
        mal("no se pudo obtener el modelo")
        traceback.print_exc()
        return 1

    # ---------------------------------------------------------------- 5
    paso(5, "Crear el FaceLandmarker")
    try:
        opciones = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ruta)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=3,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        landmarker = vision.FaceLandmarker.create_from_options(opciones)
        bien("creado con blendshapes + matrices de transformacion")
    except Exception:
        mal("no se pudo crear")
        traceback.print_exc()
        print("\n   Si el error menciona 'transformation_matrixes', prueba")
        print("   ponerlo en False en src/detector.py — es lo unico que")
        print("   agregue para la version de vigilancia.")
        return 1

    # ---------------------------------------------------------------- 6
    paso(6, "Abrir la camara")
    try:
        from src.camaras import detectar
        cams = detectar(maximo=4)
        if not cams:
            mal("ninguna camara respondio")
            print("   Cierrala en Teams, Zoom u OBS y vuelve a probar.")
            return 1
        for c in cams:
            print(f"        {c.etiqueta}")
        elegida = next((c for c in cams if c.indice == args.camara), cams[0])
        bien(f"usando {elegida.etiqueta}")

        cap = cv2.VideoCapture(elegida.indice, elegida.backend)
        ok, frame = cap.read()
        if not ok or frame is None:
            mal("abre pero no entrega imagen")
            cap.release()
            return 1
        bien(f"frame {frame.shape[1]}x{frame.shape[0]}")
    except Exception:
        mal("error con la camara")
        traceback.print_exc()
        return 1

    # ---------------------------------------------------------------- 7
    paso(7, "Procesar 30 frames (aqui suele estar el problema)")
    try:
        import time
        t0 = time.perf_counter()
        rostros_vistos = 0
        for i in range(30):
            ok, frame = cap.read()
            if not ok:
                mal(f"la camara dejo de entregar en el frame {i}")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(imagen, int((time.perf_counter() - t0) * 1000))
            if res.face_landmarks:
                rostros_vistos += 1
        dur = time.perf_counter() - t0
        bien(f"30 frames en {dur:.1f}s  ({30 / max(dur, .01):.1f} FPS)")
        print(f"        rostros detectados en {rostros_vistos}/30 frames")
        if rostros_vistos == 0:
            print("        (normal si no habia nadie frente a la camara)")
    except Exception:
        mal("crash durante el procesamiento")
        traceback.print_exc()
        cap.release()
        return 1

    # ---------------------------------------------------------------- 8
    paso(8, "Pipeline completo del proyecto")
    try:
        from src.detector import DetectorEmociones
        with DetectorEmociones(
            ruta_modelo=ruta, umbral=0.22, num_rostros=3
        ) as det:
            for i in range(15):
                ok, frame = cap.read()
                if not ok:
                    break
                rostros = det.procesar(frame, i * 40)
            bien(f"detector del proyecto OK — {len(rostros)} rostro(s) en el ultimo frame")
            for r in rostros:
                q = r.calidad
                print(f"        {r.etiqueta}  {q.ancho_px}px  "
                      f"yaw {q.yaw:+.0f}  pitch {q.pitch:+.0f}  factor {q.factor:.2f}")
    except Exception:
        mal("el detector del proyecto falla")
        traceback.print_exc()
        cap.release()
        return 1
    finally:
        cap.release()
        landmarker.close()

    print("\n" + "=" * 62)
    print("TODO CORRECTO — el pipeline funciona de principio a fin.")
    print("Si la interfaz sigue sin mostrar video, el problema esta en")
    print("app.py y no en la deteccion. Avisa con esta salida.")
    print("=" * 62 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
