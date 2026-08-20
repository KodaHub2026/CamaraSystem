"""
ui.py
=====
Dibujo sobre el frame: caja del rostro, etiqueta emocional y panel de barras.

Nota: cv2.putText usa fuentes Hershey, que NO soportan acentos ni la enie.
Por eso los textos en pantalla van sin acentos. Si necesitas acentos, hay que
dibujar con Pillow (ImageDraw.text + fuente TTF); cuesta unos FPS.
"""

from __future__ import annotations

import cv2
import numpy as np

from .emociones import COLOR_NEUTRAL, NEUTRAL, ORDEN, REGLAS

FUENTE = cv2.FONT_HERSHEY_SIMPLEX
BLANCO = (255, 255, 255)
NEGRO = (25, 25, 25)
GRIS = (170, 170, 170)


def _panel(frame, x, y, ancho, alto, alpha=0.60, color=NEGRO) -> None:
    """Rectangulo semitransparente para que el texto se lea sobre cualquier fondo."""
    x, y = max(0, x), max(0, y)
    recorte = frame[y : y + alto, x : x + ancho]
    if recorte.size == 0:
        return
    capa = np.full(recorte.shape, color, dtype=np.uint8)
    cv2.addWeighted(capa, alpha, recorte, 1 - alpha, 0, recorte)


def _barra(frame, x, y, ancho, alto, valor, color, umbral=None) -> None:
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (65, 65, 65), -1)
    relleno = int(ancho * float(np.clip(valor, 0.0, 1.0)))
    if relleno > 0:
        cv2.rectangle(frame, (x, y), (x + relleno, y + alto), color, -1)
    if umbral is not None:
        mx = x + int(ancho * umbral)
        cv2.line(frame, (mx, y - 2), (mx, y + alto + 2), BLANCO, 1)


def dibujar_rostro(frame, rostro, mostrar_malla: bool = False) -> None:
    """Caja tipo visor + etiqueta de la emocion dominante."""
    x, y, w, h = rostro.bbox
    color = rostro.color

    if mostrar_malla:
        alto_f, ancho_f = frame.shape[:2]
        for p in rostro.landmarks:
            cv2.circle(frame, (int(p.x * ancho_f), int(p.y * alto_f)), 1, (200, 200, 200), -1)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    largo = max(14, int(min(w, h) * 0.18))
    for px, py, dx, dy in (
        (x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1),
    ):
        cv2.line(frame, (px, py), (px + dx * largo, py), color, 4)
        cv2.line(frame, (px, py), (px, py + dy * largo), color, 4)

    texto = f"{rostro.etiqueta}  {rostro.emocion.confianza * 100:.0f}%"
    (tw, th), _ = cv2.getTextSize(texto, FUENTE, 0.62, 2)
    ey = max(th + 14, y - 12)
    _panel(frame, x, ey - th - 9, tw + 18, th + 16)
    cv2.putText(frame, texto, (x + 9, ey), FUENTE, 0.62, color, 2, cv2.LINE_AA)

    if rostro.emocion.dominante != NEUTRAL:
        au = rostro.emocion.au
        cv2.putText(frame, au, (x + 2, y + h + 20), FUENTE, 0.40, color, 1, cv2.LINE_AA)

    _barra(frame, x, y + h + 28, w, 8, rostro.emocion.confianza, color)


def dibujar_panel_emociones(frame, rostro, umbral: float) -> None:
    """Barras horizontales con la activacion de las 7 emociones."""
    alto_f, ancho_f = frame.shape[:2]
    px, py = ancho_f - 250, 12
    _panel(frame, px, py, 238, 26 + len(ORDEN) * 20 + 10)

    cv2.putText(frame, "ESPECTRO EMOCIONAL", (px + 12, py + 20), FUENTE, 0.44, BLANCO, 1, cv2.LINE_AA)

    puntajes = rostro.emocion.puntajes if rostro else {}
    for i, clave in enumerate(ORDEN):
        yb = py + 34 + i * 20
        valor = puntajes.get(clave, 0.0)
        activa = rostro and rostro.emocion.dominante == clave
        color = REGLAS[clave]["color"] if valor >= umbral else (95, 95, 95)

        nombre = REGLAS[clave]["etiqueta"]
        cv2.putText(frame, nombre, (px + 12, yb + 9), FUENTE, 0.36,
                    BLANCO if activa else GRIS, 2 if activa else 1, cv2.LINE_AA)
        _barra(frame, px + 108, yb + 1, 96, 9, valor, color, umbral)
        cv2.putText(frame, f"{valor:.2f}", (px + 208, yb + 9), FUENTE, 0.34, GRIS, 1, cv2.LINE_AA)


def dibujar_panel_llm(frame, interprete, alto_panel: int = 96) -> None:
    """Panel inferior con la interpretacion de Qwen y su estado."""
    if interprete is None:
        return
    alto_f, ancho_f = frame.shape[:2]
    px, py = 12, alto_f - alto_panel - 52
    _panel(frame, px, py, 470, alto_panel)

    ultima = interprete.ultima
    if interprete.ocupado:
        estado, color = "QWEN PENSANDO...", (60, 200, 250)
    elif ultima is None:
        estado, color = "QWEN EN ESPERA  (tecla I)", GRIS
    elif ultima.error:
        estado, color = "QWEN SIN CONEXION", (60, 60, 240)
    else:
        estado, color = f"QWEN  {ultima.hora}  ({ultima.segundos:.0f}s)", (150, 220, 150)

    cv2.putText(frame, estado, (px + 12, py + 20), FUENTE, 0.42, color, 1, cv2.LINE_AA)

    if ultima is None:
        cv2.putText(frame, "Las reglas FACS corren local a 30 FPS.", (px + 12, py + 42),
                    FUENTE, 0.38, GRIS, 1, cv2.LINE_AA)
        cv2.putText(frame, "Qwen agrega matiz bajo demanda.", (px + 12, py + 60),
                    FUENTE, 0.38, GRIS, 1, cv2.LINE_AA)
        return

    y = py + 42
    if ultima.matiz:
        cv2.putText(frame, ultima.matiz.upper(), (px + 12, y), FUENTE, 0.50,
                    (150, 220, 150), 2, cv2.LINE_AA)
        y += 22

    for linea in _envolver(ultima.texto, 58)[:3]:
        cv2.putText(frame, linea, (px + 12, y), FUENTE, 0.38, BLANCO, 1, cv2.LINE_AA)
        y += 16

    for i, m in enumerate(ultima.momentos[:2]):
        cv2.putText(frame, f"- {str(m)[:54]}", (px + 12, y), FUENTE, 0.36, GRIS, 1, cv2.LINE_AA)
        y += 15


def _envolver(texto: str, ancho: int) -> list:
    """Parte el texto en lineas sin cortar palabras."""
    palabras, lineas, actual = str(texto).split(), [], ""
    for w in palabras:
        if len(actual) + len(w) + 1 <= ancho:
            actual = f"{actual} {w}".strip()
        else:
            lineas.append(actual)
            actual = w
    if actual:
        lineas.append(actual)
    return lineas


def dibujar_hud(frame, rostros, umbral: float, fps: float, grabando: bool = False) -> None:
    """Panel superior izquierdo con estado general."""
    alto_f, ancho_f = frame.shape[:2]
    _panel(frame, 12, 12, 300, 76)

    cv2.putText(frame, "DETECTOR DE EMOCIONES", (24, 36), FUENTE, 0.55, BLANCO, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:4.1f}   Rostros {len(rostros)}   Umbral {umbral:.2f}",
                (24, 56), FUENTE, 0.42, GRIS, 1, cv2.LINE_AA)

    if rostros:
        r = rostros[0]
        top = ", ".join(f"{REGLAS[k]['etiqueta'][:4]} {v:.2f}" for k, v in r.emocion.top(2))
        cv2.putText(frame, top, (24, 76), FUENTE, 0.40, r.color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Sin rostro en cuadro", (24, 76), FUENTE, 0.40, GRIS, 1, cv2.LINE_AA)

    if grabando:
        cv2.circle(frame, (ancho_f - 268, 22), 6, (60, 60, 240), -1)

    ayuda = "Q salir  M malla  E espectro  +/- umbral  G captura  R registro  I interpretar  S sesion"
    (tw, _), _ = cv2.getTextSize(ayuda, FUENTE, 0.40, 1)
    _panel(frame, 12, alto_f - 40, tw + 24, 28)
    cv2.putText(frame, ayuda, (24, alto_f - 21), FUENTE, 0.40, (200, 200, 200), 1, cv2.LINE_AA)


# =========================================================================== #
# Vista de vigilancia
# =========================================================================== #


def dibujar_rostro_vigilancia(frame, rostro, malla=False, detalle_calidad=False) -> None:
    """Caja por persona con ID persistente, emocion y senales de comportamiento."""
    x, y, w, h = rostro.bbox
    fiable = rostro.fiable
    color = rostro.color

    if malla:
        af, anf = frame.shape[:2]
        for p in rostro.landmarks:
            cv2.circle(frame, (int(p.x * anf), int(p.y * af)), 1, (190, 190, 190), -1)

    # Linea punteada cuando la calidad no alcanza: senal visual inequivoca.
    if fiable:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    else:
        for i in range(x, x + w, 12):
            cv2.line(frame, (i, y), (min(i + 6, x + w), y), color, 1)
            cv2.line(frame, (i, y + h), (min(i + 6, x + w), y + h), color, 1)
        for j in range(y, y + h, 12):
            cv2.line(frame, (x, j), (x, min(j + 6, y + h)), color, 1)
            cv2.line(frame, (x + w, j), (x + w, min(j + 6, y + h)), color, 1)

    pid = getattr(rostro, "pista_id", rostro.id)
    if fiable:
        txt = f"#{pid}  {rostro.etiqueta}  {rostro.confianza_efectiva * 100:.0f}%"
    else:
        txt = f"#{pid}  {rostro.etiqueta}"

    esc = 0.44 if w < 150 else 0.54
    (tw, th), _ = cv2.getTextSize(txt, FUENTE, esc, 1)
    ey = max(th + 10, y - 8)
    _panel(frame, x, ey - th - 7, tw + 14, th + 12)
    cv2.putText(frame, txt, (x + 7, ey), FUENTE, esc, color, 1, cv2.LINE_AA)

    yb = y + h + 14
    if not fiable and rostro.calidad:
        cv2.putText(frame, rostro.calidad.resumen[:36], (x, yb), FUENTE, 0.36,
                    (130, 130, 130), 1, cv2.LINE_AA)
        yb += 14

    comp = getattr(rostro, "comportamiento", None)
    if comp is not None and comp.notas:
        cv2.putText(frame, " | ".join(comp.notas[:3])[:42], (x, yb), FUENTE, 0.36,
                    (200, 190, 120), 1, cv2.LINE_AA)
        yb += 14

    if detalle_calidad and rostro.calidad:
        q = rostro.calidad
        cv2.putText(frame, f"{q.ancho_px}px y{q.yaw:+.0f} p{q.pitch:+.0f} q{q.factor:.2f}",
                    (x, yb), FUENTE, 0.34, (140, 170, 190), 1, cv2.LINE_AA)


def dibujar_hud_vigilancia(frame, rostros, agregador, fuente, umbral, fps) -> None:
    """Panel de operacion: aforo, calidad, alertas y estado del stream."""
    af, anf = frame.shape[:2]
    _panel(frame, 12, 12, 320, 116)

    fiables = [r for r in rostros if r.fiable]
    res = agregador.resumen_ventana()

    cv2.putText(frame, "VIGILANCIA EMOCIONAL", (24, 34), FUENTE, 0.52, BLANCO, 1, cv2.LINE_AA)

    est = fuente.estado
    color_est = (150, 220, 150) if est.conectada else (60, 60, 240)
    cv2.putText(frame, "STREAM OK" if est.conectada else "SIN SENAL",
                (232, 34), FUENTE, 0.38, color_est, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Aforo {len(rostros)}   Fiables {len(fiables)}   {fps:4.1f} FPS",
                (24, 56), FUENTE, 0.40, GRIS, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Predominante: {res['dominante']}", (24, 76),
                FUENTE, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    q = res["calidad_media"]
    cq = (90, 220, 120) if q > 0.6 else ((60, 200, 250) if q > 0.35 else (60, 60, 240))
    cv2.putText(frame, "Calidad", (24, 96), FUENTE, 0.38, GRIS, 1, cv2.LINE_AA)
    _barra(frame, 86, 88, 110, 8, q, cq, 0.35)
    cv2.putText(frame, f"{q:.2f}", (204, 96), FUENTE, 0.36, cq, 1, cv2.LINE_AA)

    desc = agregador.descartadas_calidad
    tot = max(agregador.total_lecturas, 1)
    cv2.putText(frame, f"Descartadas {desc * 100 // tot}%", (238, 96),
                FUENTE, 0.36, GRIS, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Umbral {umbral:.2f}   Reconex. {est.reconexiones}",
                (24, 116), FUENTE, 0.36, (140, 140, 140), 1, cv2.LINE_AA)

    # Alertas recientes, esquina superior derecha
    if agregador.alertas:
        recientes = list(agregador.alertas)[-3:]
        _panel(frame, anf - 292, 12, 280, 20 + len(recientes) * 18, alpha=0.7)
        cv2.putText(frame, "ALERTAS", (anf - 280, 30), FUENTE, 0.40,
                    (60, 60, 240), 1, cv2.LINE_AA)
        for i, a in enumerate(recientes):
            cv2.putText(frame, f"{a.hora} {a.tipo} {a.detalle}"[:40],
                        (anf - 280, 48 + i * 18), FUENTE, 0.34, (200, 200, 200), 1, cv2.LINE_AA)

    ayuda = "Q salir  M malla  C calidad  +/- umbral  G captura  I interpretar  S sesion"
    (tw, _), _ = cv2.getTextSize(ayuda, FUENTE, 0.38, 1)
    _panel(frame, 12, af - 38, tw + 22, 26)
    cv2.putText(frame, ayuda, (23, af - 20), FUENTE, 0.38, (195, 195, 195), 1, cv2.LINE_AA)
