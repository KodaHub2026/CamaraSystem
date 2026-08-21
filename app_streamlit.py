"""
app_streamlit.py
================
Versión web de Vigilancia Emocional con Streamlit + streamlit-webrtc.
La cámara se captura desde el navegador (funciona en Docker).
"""

import streamlit as st
import cv2
import av
import numpy as np

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from src.detector import DetectorEmociones
from src.ui import dibujar_rostro
from src.llm import URL_POR_DEFECTO, ClienteLMStudio


st.set_page_config(
    page_title="Vigilancia Emocional — KodaHub",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos oscuros
st.markdown("""
<style>
    .stApp { background-color: #1A1F2B; color: #E8ECF2; }
    section[data-testid="stSidebar"] { background-color: #222836; }
    h1, h2, h3 { color: #E8ECF2 !important; }
</style>
""", unsafe_allow_html=True)


class ProcesadorEmociones(VideoProcessorBase):
    def __init__(self):
        self.detector = DetectorEmociones(num_rostros=4, umbral=0.22)
        self.umbral = 0.22
        self._timestamp = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        self.detector.umbral = self.umbral
        self._timestamp += 33
        rostros = self.detector.procesar(img, self._timestamp)

        for rostro in rostros:
            dibujar_rostro(img, rostro)

        # HUD
        cv2.rectangle(img, (8, 8), (300, 42), (20, 20, 20), -1)
        cv2.putText(
            img,
            f"Rostros: {len(rostros)}  |  Umbral: {self.umbral:.2f}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 180),
            1,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("Vigilancia Emocional — KodaHub")
    st.caption("Análisis de emociones en tiempo real · Cámara del navegador")

    # Sidebar
    with st.sidebar:
        st.header("Configuración")

        umbral = st.slider("Umbral de confianza", 0.10, 0.50, 0.22, 0.01)

        st.divider()
        st.subheader("Modelo LLM")

        if st.button("Actualizar modelos"):
            with st.spinner("Consultando modelos..."):
                try:
                    cliente = ClienteLMStudio(base_url=URL_POR_DEFECTO)
                    modelos = [m for m in cliente.listar_modelos() if "embed" not in m.lower()]
                    st.session_state.modelos = modelos
                    st.success(f"{len(modelos)} modelos encontrados")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.modelos = []

        modelos = st.session_state.get("modelos", [])
        if modelos:
            st.selectbox("Modelo", modelos)
        else:
            st.caption("Presiona 'Actualizar modelos'")

        st.checkbox("Activar interpretación LLM", value=False)

        st.divider()
        st.markdown("""
        **Uso**
        1. Haz clic en **START**
        2. Permite el acceso a la cámara
        3. El análisis se dibuja sobre el video
        """)

    # Video con cámara del navegador
    ctx = webrtc_streamer(
        key="vigilancia-emocional",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=ProcesadorEmociones,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Actualizar umbral en tiempo real
    if ctx.video_processor:
        ctx.video_processor.umbral = umbral


if __name__ == "__main__":
    main()