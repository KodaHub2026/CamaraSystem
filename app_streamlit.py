"""
app_streamlit.py
================
Versión web de Vigilancia Emocional usando Streamlit.
"""

import streamlit as st
import cv2
import time

from src.motor import Configuracion, MotorVigilancia
from src.camaras import detectar
from src.llm import URL_POR_DEFECTO, ClienteLMStudio


st.set_page_config(
    page_title="Vigilancia Emocional — KodaHub",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos oscuros tipo KodaHub
st.markdown("""
<style>
    .stApp {
        background-color: #1A1F2B;
        color: #E8ECF2;
    }
    section[data-testid="stSidebar"] {
        background-color: #222836;
    }
    h1, h2, h3 {
        color: #E8ECF2 !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("Vigilancia Emocional — KodaHub")
    st.caption("Análisis de emociones en tiempo real · Versión Web")

    # Estado de la sesión
    if "motor" not in st.session_state:
        st.session_state.motor = None
    if "corriendo" not in st.session_state:
        st.session_state.corriendo = False
    if "camaras" not in st.session_state:
        st.session_state.camaras = []
    if "modelos" not in st.session_state:
        st.session_state.modelos = []

    # Sidebar de configuración
    with st.sidebar:
        st.header("Configuración")

        # Cámaras
        if st.button("Buscar cámaras"):
            with st.spinner("Detectando cámaras..."):
                st.session_state.camaras = detectar()

        camaras = st.session_state.camaras
        if camaras:
            opciones = {c.etiqueta: c for c in camaras}
            seleccion = st.selectbox("Cámara", list(opciones.keys()))
            camara_obj = opciones[seleccion]
            indice_camara = camara_obj.indice
            backend_camara = camara_obj.backend
        else:
            st.info("Presiona 'Buscar cámaras'")
            indice_camara = 0
            backend_camara = None

        st.divider()

        # Modelos (LLM)
        st.subheader("Modelo LLM")
        if st.button("Actualizar modelos"):
            with st.spinner("Consultando modelos en LM Studio..."):
                try:
                    cliente = ClienteLMStudio(base_url=URL_POR_DEFECTO)
                    modelos = cliente.listar_modelos()
                    # Filtrar embeddings
                    modelos = [m for m in modelos if "embed" not in m.lower()]
                    st.session_state.modelos = modelos
                    st.success(f"{len(modelos)} modelos encontrados")
                except Exception as e:
                    st.error(f"Error al cargar modelos: {e}")
                    st.session_state.modelos = []

        modelos = st.session_state.modelos
        if modelos:
            modelo_seleccionado = st.selectbox("Modelo", modelos)
        else:
            st.caption("Presiona 'Actualizar modelos' (necesita LM Studio / túnel activo)")
            modelo_seleccionado = ""

        llm_activo = st.checkbox("Activar interpretación LLM", value=False)

        st.divider()

        # Parámetros
        ancho = st.selectbox("Resolución ancho", [640, 1280, 1920], index=1)
        alto = st.selectbox("Resolución alto", [480, 720, 1080], index=1)
        umbral = st.slider("Umbral de confianza", 0.10, 0.50, 0.22, 0.01)
        aforo = st.slider("Aforo máximo", 1, 15, 6)
        espejo = st.checkbox("Espejo", value=True)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            iniciar = st.button("▶ INICIAR", use_container_width=True, type="primary")
        with col2:
            detener = st.button("⏹ DETENER", use_container_width=True)

    # Lógica de inicio / detención
    if iniciar and not st.session_state.corriendo:
        config = Configuracion(
            origen=indice_camara,
            tipo="usb",
            ancho=ancho,
            alto=alto,
            aforo=aforo,
            umbral=umbral,
            espejo=espejo,
            backend=backend_camara,
            llm_activo=llm_activo,
            llm_url=URL_POR_DEFECTO,
            llm_modelo=modelo_seleccionado if modelo_seleccionado else "",
        )
        motor = MotorVigilancia(config)
        motor.iniciar()
        st.session_state.motor = motor
        st.session_state.corriendo = True
        st.rerun()

    if detener and st.session_state.corriendo:
        if st.session_state.motor:
            st.session_state.motor.detener()
        st.session_state.motor = None
        st.session_state.corriendo = False
        st.rerun()

    # Área principal de video
    placeholder = st.empty()

    if st.session_state.corriendo and st.session_state.motor:
        motor = st.session_state.motor
        st.success("Sistema en ejecución")

        while st.session_state.corriendo:
            frame = motor.tomar_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            time.sleep(0.03)
    else:
        placeholder.info("Presiona **INICIAR** para comenzar el análisis")


if __name__ == "__main__":
    main()