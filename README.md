# Vigilancia Emocional

Reconoce en tiempo real las **siete emociones básicas** de una persona a partir de
sus expresiones faciales. Corre en CPU, sin GPU y sin internet (tras la primera
ejecución).

| Emoción | Action Units (FACS) | Qué se mide |
|---|---|---|
| Felicidad | AU6 + AU12 | Mejillas elevadas + comisuras arriba |
| Tristeza | AU1 + AU4 + AU15 | Ceja interna arriba + comisuras abajo |
| Enojo | AU4 + AU5 + AU7 + AU23 | Cejas abajo + párpados tensos + labios apretados |
| Sorpresa | AU1 + AU2 + AU5 + AU26 | Cejas arriba + ojos abiertos + mandíbula caída |
| Miedo | AU1+2+4 + AU5 + AU20 + AU26 | Sorpresa + labios estirados + cejas juntas |
| Asco | AU9 + AU10 | Nariz arrugada + labio superior elevado |
| Desprecio | AU12 + AU14 unilateral | **Asimetría**: una sola comisura arriba |

Más `NEUTRAL` cuando ninguna supera el umbral.

---

## Instalación

```bash
code vigilancia-emocional

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
python main.py
```

En VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → elige `.venv`.
Luego basta **F5** (ya hay tres configuraciones en `.vscode/launch.json`).

La primera ejecución descarga `face_landmarker.task` (~3.7 MB) a `modelos/`.
De ahí en adelante funciona offline.

---

## Uso

```bash
python main.py                       # cámara por defecto
python main.py --camara 1            # segunda cámara
python main.py --umbral 0.16         # más sensible
python main.py --rostros 4           # varias personas
python main.py --registro sesion.csv # graba la sesión a CSV
python main.py --malla               # ver los 478 puntos faciales
python main.py --llm                 # + interpretación con Qwen
python main.py --llm --llm-auto 20   # interpreta solo cada 20 s
```

| Tecla | Acción |
|---|---|
| `Q` / `ESC` | Salir |
| `M` | Malla facial |
| `E` | Espectro emocional |
| `+` / `-` | Ajustar umbral en vivo |
| `G` | Captura a `capturas/` |
| `R` | Iniciar / detener registro CSV |
| `I` | Interpretar el instante con Qwen (requiere `--llm`) |
| `S` | Resumen de la sesión con Qwen |

---

## Por qué reglas FACS y no un CNN entrenado

La alternativa obvia era DeepFace o FER, que traen un clasificador entrenado en
FER2013. Se descartó por cuatro razones:

1. **Precisión real.** FER2013 tiene etiquetas ruidosas; los modelos entrenados
   ahí rondan 65–70 % en las siete clases. No es una mejora clara sobre reglas
   bien calibradas.
2. **Peso.** Arrastra TensorFlow y descarga cientos de MB. Aquí todo son 3.7 MB.
3. **Velocidad.** Un CNN por frame no da tiempo real en CPU; habría que correrlo
   cada N frames y la etiqueta se siente rezagada.
4. **Interpretabilidad.** Este es el punto decisivo. Cuando el sistema dice
   "enojo", puedes ver **exactamente** qué lo activó: `browDown 0.75`,
   `eyeSquint 0.50`. Un CNN es una caja negra — si falla, no hay nada que
   ajustar. Aquí abres `emociones.py` y cambias un peso.

El mapeo no es inventado: viene de **EMFACS** (Ekman & Friesen), el estándar en
psicología de la expresión facial. Los 52 blendshapes de MediaPipe son en la
práctica aproximaciones de las Action Units de FACS.

---

## Los tres pares confusos

Aquí está el trabajo fino del clasificador.

**Miedo vs Sorpresa.** Los dos abren ojos y mandíbula — es la confusión clásica,
también en humanos. Los separa AU20 (labios estirados en horizontal) y AU4
(cejas juntas), presentes **solo** en miedo. Por eso `sorpresa` penaliza esas dos
señales con peso alto. Prueba simulada: en cara de sorpresa gana `sorpresa 0.71`
sobre `miedo 0.50`; en cara de miedo, `miedo 0.66` contra `sorpresa 0.01`.

**Enojo vs Asco.** Ambos arrugan la nariz. Domina AU4 (cejas abajo) en enojo y
AU9+AU10 en asco, así que enojo penaliza el labio superior elevado.

**Desprecio vs Felicidad.** La clave es la asimetría. Una sonrisa sube las dos
comisuras; el desprecio, una sola. Como MediaPipe entrega blendshape izquierdo y
derecho por separado, se mide directo con `_asimetriaSonrisa`. La penalización
`_sonrisaSimetrica` evita que una sonrisa normal se marque como desprecio.

---

## Suavizado e histéresis

Dos capas para que la etiqueta no parpadee:

1. **Promedio móvil** de 8 frames sobre cada emoción (`--suavizado`).
2. **Histéresis**: la emoción actual solo cede si otra la supera por `0.06`
   durante 3 frames seguidos.

Sin la segunda capa, en una transición gradual la etiqueta rebota decenas de
veces entre las dos emociones vecinas. Con ella, la transición
felicidad → sorpresa produce **un solo cambio limpio**.

---

## Calibración

El umbral por defecto es `0.22`. La expresividad facial varía muchísimo entre
personas, así que ajústalo en vivo con `+` / `-`:

- **Baja a 0.14–0.18** con gente poco expresiva.
- **Sube a 0.30–0.40** si marca emociones con cara neutral.

La **iluminación** importa tanto como el umbral: luz frontal difusa da lecturas
mucho más estables que contraluz o una ventana atrás.

### Confiabilidad por emoción

No todas salen igual de bien. Con honestidad:

- **Muy confiables** — felicidad, sorpresa. AUs grandes e inconfundibles.
- **Confiables** — enojo, asco, tristeza. Necesitan expresión marcada.
- **La menos confiable** — desprecio. Depende de asimetría, y girar la cabeza
  produce asimetría aparente. Si vas a usarlo en producción, pide que la persona
  mire de frente, o quítalo de `REGLAS`.

---

## Capa 2: interpretación con Qwen (LM Studio)

```bash
python verificar_lm.py          # diagnostica el túnel primero
python main.py --llm            # activa la interpretación
```

Con `--llm`, la tecla `I` pide a Qwen que interprete el momento actual y `S`
genera un resumen de la sesión completa.

### Por qué el LLM no puede clasificar frame por frame

Esta fue la decisión de arquitectura más importante del proyecto, y está basada
en mediciones reales contra `lmmstudio5090.koda-cloud.org`:

| Componente | Latencia | Frecuencia posible |
|---|---|---|
| MediaPipe + reglas FACS | **33 ms** | 30 FPS |
| `qwen/qwen3.8-27b` | **78 s** | ~1 por minuto |

Son tres órdenes de magnitud. No es un problema de optimización: un frame dura
33 ms y el modelo tarda 78,000 ms. Por eso el sistema tiene dos capas con
responsabilidades distintas:

| | Capa 1 (local) | Capa 2 (Qwen) |
|---|---|---|
| Responde | ¿Qué músculos se mueven? | ¿Qué significa el patrón? |
| Salida | 7 categorías fijas | Matices, mezclas, narrativa |
| Bloquea el video | Nunca | Nunca (hilo aparte) |

Ejemplo real de la diferencia. Ante la misma secuencia, las reglas dieron
`sorpresa 0.71`. Qwen devolvió:

```json
{"matiz": "sorpresa tensa",
 "lectura": "Neutral inicial que se tensa en sorpresa incomoda.",
 "confianza": "media"}
```

El LLM capturó el **arco temporal**, algo que una clasificación por frame no
puede expresar.

### Tres problemas reales del túnel (medidos, no teóricos)

**1. Cloudflare corta a los ~100 s → HTTP 524.**
Sin streaming, la petición muere con `error code: 524` a los 125 s. No es un
timeout del cliente: es el límite de origen de Cloudflare. Con `stream: true`
los tokens fluyen, Cloudflare ve actividad y nunca dispara el error. Medido:

```
sin stream  ->  HTTP 524 a los 125 s
con stream  ->  HTTP 200 a los 238 s
```

Aplica a **todos** los túneles de KodaHub, no solo a este proyecto.

**2. `qwen3.8-27b` es un modelo de razonamiento.**
Con `max_tokens=180` devolvió contenido **vacío**: los 180 tokens se consumieron
en `reasoning_tokens`. Por eso `MAX_TOKENS = 3000`. Bajarlo "para acelerar" es
justo lo que rompe la respuesta.

**3. Emite bloques `<think>` y separa `reasoning_content`.**
El parser los descarta y además **corta el stream apenas el JSON está completo**
— sin ese corte, la latencia sube de 78 s a 238 s generando texto inútil.

### Recomendación

`qwen3.8-27b` funciona, pero 78 s es mucho. En tu servidor hay 29 modelos
listados; casi todos responden *"model not loaded"* porque el **JIT loading está
apagado**. Vale la pena cargar uno más ágil:

```bash
python verificar_lm.py --modelo google/gemma-4-12b
python verificar_lm.py --modelo qwen/qwen3-vl-8b
```

Un 8B sin razonamiento debería bajar a 3–8 s, suficiente para interpretación
casi continua con `--llm-auto 15`.

> Si el túnel está caído, todo lo demás sigue funcionando. MediaPipe y las reglas
> FACS son 100 % locales; el LLM es opcional por diseño.

---

## Análisis de sesiones

Con `--registro` o la tecla `R` se genera un CSV con una fila por frame:

```
timestamp, segundos, rostro, dominante, confianza, felicidad, tristeza, enojo, ...
```

Sirve para medir reacciones a lo largo del tiempo — un video, una entrevista, una
demo de producto:

```python
import pandas as pd
df = pd.read_csv("registros/sesion_20260819_143022.csv")
df.groupby("dominante").size()                    # tiempo por emoción
df.set_index("segundos")[["felicidad","sorpresa"]].plot()
```

---

## Estructura

```
deteccion-emociones/
├── main.py                  # bucle de captura y controles
├── requirements.txt
├── src/
│   ├── emociones.py         # reglas FACS + histéresis   ← el núcleo
│   ├── detector.py          # MediaPipe Face Landmarker
│   ├── ui.py                # HUD y espectro emocional
│   ├── llm.py               # capa 2: Qwen asíncrono + streaming
│   └── registro.py          # exportación a CSV
├── verificar_lm.py          # diagnóstico del túnel LM Studio
├── modelos/                 # face_landmarker.task (se descarga solo)
├── capturas/
├── registros/
└── .vscode/launch.json
```

Para ajustar el comportamiento, casi siempre vas a tocar el diccionario `REGLAS`
en `src/emociones.py`. Cada emoción lista sus señales con peso y sus
penalizaciones; no hace falta entender el resto del código.

---

## Problemas comunes

**La cámara no abre.** Ciérrala en Teams / Zoom / OBS y prueba `--camara 1`.

**Acentos como `?` en pantalla.** OpenCV usa fuentes Hershey sin soporte de
acentos. Por eso el HUD va sin ellos. Para acentos hay que dibujar con Pillow.

**Va lento.** Baja resolución: `--ancho 640 --alto 480`. En un i5 moderno
deberías ver 25–30 FPS a 720p.

**`mediapipe` no instala.** Requiere Python 3.9–3.12; en 3.13 aún no hay wheels
estables.

**Detecta emociones raras con cara neutral.** Sube el umbral con `+`. Algunas
caras en reposo tienen las comisuras naturalmente hacia abajo y disparan tristeza.


---

## Interfaz gráfica

```bash
python app.py
```

**La cámara no se abre al arrancar.** Configuras fuente, modelo y parámetros; el
análisis empieza cuando presionas INICIAR. Para operación desde terminal o modo
desatendido, `main.py` sigue disponible con todos sus flags.

### Selector de modelo con prueba real

El desplegable se llena desde `/v1/models` del túnel, pero **aparecer en la lista
no significa estar cargado**: en el servidor de KodaHub había 29 modelos
listados y solo uno respondía, porque el JIT loading está apagado.

Por eso hay un botón **Probar modelo** que mide tres cosas que la lista no dice:

| Resultado | Significado |
|---|---|
| Verde, `8s` | Cargado y ágil |
| Ámbar, `30s razona` | Usable, pero gasta tokens razonando |
| Naranja, `78s` | Solo para consultas bajo demanda |
| Rojo, `no cargado` | Está listado pero no en memoria |

La prueba corre en un hilo aparte con timeout de 100 s — el límite de
Cloudflare. Si un modelo no responde en ese margen, tampoco va a servir en
producción.

### Por qué Tkinter y no Qt o web

Viene incluido con Python en Windows: cero instalación extra, cero dependencias
pesadas, arranque instantáneo. Para un panel que se abre antes de una demo,
eso vale más que un framework bonito que hay que instalar en cada máquina.

El video se pinta dentro de la ventana con Pillow, pero **se analiza a
resolución completa y se muestra reducido** — hacen falta píxeles de rostro
para el portero de calidad, no para el display.

### Arquitectura de hilos

Tkinter no es thread-safe: todo widget se toca solo desde el hilo principal. Y
el análisis consume 20–35 ms por frame; en el hilo de la GUI, Windows marcaría
la ventana como «no responde».

```
hilo motor  ->  captura, analiza, deja el frame en una ranura protegida
hilo GUI    ->  cada 33 ms lee la ranura y pinta
hilos LLM   ->  resultados llegan por cola, se aplican en el tick
```

`src/motor.py` no sabe que existe una interfaz. Expone estado y frames; quien lo
consuma decide cómo mostrarlos.

---

## Modo vigilancia

```bash
python main.py --rtsp "rtsp://usuario:clave@10.0.0.50:554/stream1" --aforo 8
python main.py --rtsp "..." --sin-ventana --salida metricas.csv    # desatendido
python main.py --video grabacion.mp4 --aforo 6                     # analizar archivo
```

| Flag | Para qué |
|---|---|
| `--rtsp URL` | Cámara IP. Reconecta sola con retroceso exponencial |
| `--aforo N` | Personas simultáneas (default 6) |
| `--saltar N` | Analiza 1 de cada N frames. Duplica el aforo posible |
| `--sin-ventana` | Desatendido, sin GUI. Loguea a stdout |
| `--salida CSV` | Métricas agregadas por ventana |
| `--alerta-aforo N` | Alerta con N o más personas |
| `--alerta-seg S` | Segundos sostenidos antes de alertar (default 4) |

### El portero de calidad: lo más importante del sistema

Una cámara de vigilancia entrega justo lo contrario de lo que MediaPipe
necesita:

| Webcam | Vigilancia |
|---|---|
| rostro 350 px | rostro 35–80 px |
| frontal | montada en alto, mira hacia abajo |
| luz de pantalla | contraluz, IR nocturno |
| 1 persona | varias, a distintas distancias |

El problema no es que baje la precisión. Es que **MediaPipe sigue devolviendo
números con la misma pinta de confianza**. Un rostro de 30 px produce
blendshapes que son ruido, y sin portero el sistema reportaría `ENOJO 0.58` con
total aplomo.

Peor: el pitch negativo (cámara mirando hacia abajo) comprime verticalmente la
zona de las cejas, y el modelo lo lee como `browDown` → **falso enojo
sistemático**. Ese sesgo es consistente, no aleatorio, así que promediar frames
no lo corrige: lo consolida.

`calidad.py` mide resolución, ángulo, nitidez y luz, y produce un factor 0–1 que
multiplica la confianza. Por debajo de 0.35 el sistema reporta
`CALIDAD INSUFICIENTE` y dibuja la caja punteada en vez de inventar una emoción.
Comportamiento medido:

| Escenario | Factor | Acción |
|---|---|---|
| Webcam ideal (350px frontal) | 1.00 | reporta |
| Vigilancia buena (140px) | 1.00 | reporta |
| Vigilancia típica (75px, ángulo alto) | 0.39 | reporta con aviso |
| Rostro lejano (38px) | 0.00 | **bloquea** |
| Cámara muy alta (pitch −42°) | 0.00 | **bloquea** |
| Perfil (yaw 58°) | 0.00 | **bloquea** |
| Movimiento / desenfoque | 0.00 | **bloquea** |

El factor es **multiplicativo** a propósito: si el rostro mide 30 px, no importa
que la luz sea perfecta.

### Seguimiento y comportamiento

`seguimiento.py` asigna IDs persistentes por IoU. Sin esto, la persona A
heredaría el historial emocional de la persona B en el frame siguiente, porque
MediaPipe no garantiza el orden de los rostros.

Sobre la trayectoria se derivan señales que el rostro solo no da:
**permanencia**, **agitación** (variabilidad de rotación de cabeza),
**inquietud** (desplazamiento del centro), **acercamiento** (el rostro crece o
encoge) y **cabeza baja** (pitch sostenido).

Todo sale del pose de cabeza y la caja, sin modelos extra. Para postura corporal
completa haría falta MediaPipe Pose Landmarker, que en multipersona a distancia
cuesta más computo del que aporta aquí.

### Agregado por defecto, no identificación

`agregado.py` **no guarda rostros, ni embeddings, ni nada reidentificable**. Los
IDs son efímeros: mueren cuando la persona sale de cuadro, y reiniciar el
programa reinicia la numeración. Lo que persiste son conteos por ventana.

No es una restricción arbitraria. Un sistema que sigue individuos entre sesiones
es un sistema de identificación biométrica, con obligaciones legales mucho más
pesadas. En agregado el sistema mide *el ambiente de la sala* en vez de *el
estado de Fulano* — que además es para lo que la tecnología realmente alcanza.

Las alertas exigen persistencia (`--alerta-seg`, default 4 s) porque una lectura
aislada de enojo en un frame no significa nada.

---

## Antes de instalarlo en un sitio real

Tres cosas que conviene resolver antes que el código.

**1. Marco legal.** En México, los datos biométricos son datos personales
sensibles bajo la LFPDPPP: requieren consentimiento **expreso** y aviso de
privacidad específico. No basta con un letrero de "zona videovigilada", que
cubre grabación pero no inferencia biométrica. El AI Act europeo va más lejos y
**prohíbe** el reconocimiento de emociones en centros de trabajo y educativos —
relevante dado el trabajo de KodaHub con la UTM. Aunque no aplique jurisdicción
europea, es una señal fuerte sobre dónde está aterrizando la regulación.

**2. Validez científica.** El sistema mide **qué músculos se mueven**, no qué
siente la persona. La investigación de Lisa Feldman Barrett muestra que esa
relación es más débil y más dependiente de contexto y cultura de lo que asumía
el modelo clásico de Ekman. A distancia de vigilancia, con ángulo picado y
rostros de 60 px, esa incertidumbre se multiplica.

**3. Dónde sí funciona bien.** Métricas agregadas de audiencia (un stand, una
sala), disparo de contenido en señalización digital, conteo de aforo con
contexto emocional, análisis de reacción a material audiovisual con
participantes que consintieron.

**Dónde no.** Evaluación de personas, control de acceso, detección de
intenciones, cualquier decisión individual con consecuencias. Ahí no hay
validez, y el sesgo de falso enojo por ángulo picado convertiría a cualquiera
que mire hacia abajo en sospechoso.

Si el resumen final reporta más de 45 % de lecturas descartadas, la instalación
física está mal: revisa altura, ángulo e iluminación antes de dar peso a los
números.

---

## Advertencia de uso

El reconocimiento de emociones por expresión facial mide **qué músculos se
mueven**, no qué siente la persona. La investigación de Lisa Feldman Barrett y
otros muestra que la relación entre expresión y estado interno es más débil y más
dependiente de contexto y cultura de lo que asumía el modelo clásico de Ekman.

En la práctica: sirve muy bien para reaccionar a expresiones (avatares, juegos,
métricas agregadas de audiencia, accesibilidad). No lo uses para decidir sobre
personas — contratación, evaluación, detección de mentiras. Ahí no tiene validez.
