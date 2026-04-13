import streamlit as st

st.set_page_config(
    page_title="☕ Coffee Quality - Henry Bootcamp",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Coffee Quality Predictor & Recommendation System")
st.markdown("---")

st.markdown("""
## 🎉 Bienvenido al Sistema Integral de Calidad de Café

Este proyecto es el resultado del trabajo colaborativo del **Bootcamp de Data Science Henry**, 
donde diferentes desarrolladores contribuyeron con sus modelos especializados y funcionalidades innovadoras.

### 🏗️ **Arquitectura del Sistema**

Nuestra plataforma está organizada en múltiples páginas especializadas, cada una con un propósito específico:

---

### 📋 **Páginas Disponibles**

#### 🌲 **Random Forest** — Modelo de Felipe Baquero
- **Autor:** Felipe Baquero
- **Características:** Modelo especializado Random Forest con análisis completo de características
- **Funcionalidades:** Predicción de calidad, análisis de importancia de variables

#### 🌿 **Gradient Boosting** — Modelo de Matías Gutierrez  
- **Autor:** Matías Gutierrez
- **Características:** Modelo especializado Gradient Boosting optimizado
- **Funcionalidades:** Predicción de calidad, sistema de recomendación, análisis del modelo

#### 🏆 **Best Model** — Sistema Dinámico de Matias Gutierrez y Felipe Baquero
- **Autor:** Matias Gutierrez y Felipe Baquero
- **Características:** Sistema inteligente que selecciona automáticamente el mejor modelo según rendimiento
- **Funcionalidades:** 
  - 🎯 Predicción con el mejor modelo (actualmente SVR)
  - 🔍 Sistema de recomendación avanzado
  - 📊 Análisis comparativo de todos los modelos
  - 🎮 **Demo Interactivo** — Integración de demos de predicción y recomendación

---

### 🎮 **Sistema de Demo Interactivo**

Desarrollado por **Enzo Zambón**, esta sección revoluciona la experiencia del usuario al integrar:

- **🎯 Demo de Predicción Interactiva:** Transforma el demo de consola en una experiencia visual completa
- **🔍 Demo de Recomendación Interactiva:** Convierte el sistema de recomendación en una interfaz dinámica
- **📊 Visualizaciones en tiempo real:** Gráficos interactivos y análisis exploratorio

---

### 🚀 **¿Cómo funciona?**

1. **Selecciona una página** desde el menú lateral izquierdo
2. **Explora las funcionalidades** específicas de cada modelo
3. **Compara resultados** entre diferentes enfoques
4. **Experimenta con los demos** interactivos para entender el funcionamiento

---

### 📈 **Innovaciones Implementadas**

- ✅ **Modelado Dinámico:** Selección automática del mejor modelo
- ✅ **Integración de Demos:** Transformación de demos de consola a interfaces interactivas  
- ✅ **Análisis Comparativo:** Visualización de rendimiento entre modelos
- ✅ **Sistema de Recomendación:** Búsqueda inteligente basada en similitud
- ✅ **Visualizaciones Avanzadas:** Gráficos interactivos con Plotly

---

### 👥 **Equipo de Desarrollo**

**Data Science Henry Bootcamp - Proyecto Final**

- **🌲 Felipe Baquero:** Random Forest Specialist y Best Model
- **🌿 Matías Gutierrez:** Gradient Boosting Developer y Best Model 
- **🏆 Enzo Zambón:** Interactive Systems Architect

*Un proyecto colaborativo que demuestra el poder del trabajo en equipo y la aplicación de diferentes técnicas de Machine Learning.*
""")

st.info("👈 Seleccioná una página en el sidebar para comenzar a explorar el sistema completo.")

st.markdown("---")

# Sección de características destacadas
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🏆 **Best Model**
    **Sistema inteligente que siempre usa el modelo con mejor rendimiento.**
    
    🎯 *Predicción dinámica*  
    🔍 *Recomendación avanzada*  
    📊 *Análisis comparativo*  
    🎮 *Demos interactivos*
    """)

with col2:
    st.markdown("""
    ### 🌿 **Gradient Boosting**  
    **Modelo especializado con optimización específica.**
    
    🎯 *Predicción precisa*  
    🔍 *Sistema de recomendación*  
    📊 *Análisis del modelo*  
    ⚙️ *Configuración avanzada*
    """)

with col3:
    st.markdown("""
    ### 🌲 **Random Forest**
    **Modelo clásico con análisis robusto.**
    
    🎯 *Predicción confiable*  
    📊 *Análisis de características*  
    🔍 *Importancia de variables*  
    ⚙️ *Interpretación de resultados*
    """)

st.markdown("---")

st.markdown("""
### 🎯 **Recomendación de Uso**

- **🔰 Principiantes:** Comenzar con **Best Model** para explorar todas las funcionalidades
- **⚙️ Avanzados:** Comparar resultados entre **Best Model** y **Gradient Boosting**  
- **📊 Análisis:** Usar **Random Forest** para entender la importancia de variables
- **🎮 Experimentación:** Probar los **demos interactivos** en **Best Model**

---

### 📞 **Soporte y Desarrollo**

Este sistema representa la culminación del Bootcamp de Data Science Henry,
demostrando la capacidad de integrar múltiples modelos y crear una plataforma
unificada para el análisis de calidad de café.
""")
