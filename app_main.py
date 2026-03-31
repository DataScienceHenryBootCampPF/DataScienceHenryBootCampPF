import streamlit as st

st.set_page_config(
    page_title="☕ Coffee Quality - Henry Bootcamp",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Coffee Quality Predictor")
st.markdown("---")

st.markdown("""
## Bienvenido al sistema de predicción de calidad de café

Este proyecto compara dos modelos de Machine Learning para predecir 
la calidad del café usando el dataset de Coffee Quality Institute.

### 📌 Navegá por las secciones:

- 🌲 **Random Forest** — Modelo de Felipe Baquero
- 🚀 **Gradient Boosting** — Modelo de Matias Gutierrez

Usá el menú de la izquierda para explorar cada modelo.
""")

st.info("👈 Seleccioná una página en el sidebar para comenzar.")