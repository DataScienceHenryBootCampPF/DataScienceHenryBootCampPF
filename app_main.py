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
- **🧪 Claudia Rivero:** Quality Assurance & System Testing

*Un proyecto colaborativo que demuestra el poder del trabajo en equipo y la aplicación de diferentes técnicas de Machine Learning.*
""")

st.info("👈 Seleccioná una página en el sidebar para comenzar a explorar el sistema completo.")

# Opción adicional para ver la guía de calidad
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Recursos")
show_guide = st.sidebar.checkbox("📖 Mostrar Guía de Calidad de Cafés")

if show_guide:
    st.sidebar.markdown("""
    ### 📖 **Guía de Calidad de Cafés**
    
    **¿Qué encontrarás?**
    - 🎯 **Criterios de calidad** para cada modelo
    - 📊 **Umbrales** de clasificación (Premium/Estándar)
    - 🔍 **Factores clave** que determinan la calidad
    - 📈 **Estrategias** para mejorar de estándar a premium
    - 💡 **Casos prácticos** para productores y tostadores
    
    **Modelos Analizados:**
    - 🌿 **Gradient Boosting** (Matías Gutierrez)
    - 🌲 **Random Forest** (Felipe Baquero)
    - 🏆 **Best Model** (SVR - Sistema dinámico)
    
    **Ubicación:** `pages/Guia de calidad de cafes.md`
    """)
    
    # Mostrar contenido de la guía
    try:
        with open("pages/Guia de calidad de cafes.md", "r", encoding="utf-8") as f:
            guide_content = f.read()
        
        st.markdown("---")
        st.header("📖 Guía de Calidad de Cafés")
        
        # Botones de navegación rápida
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎯 Umbrales", key="thresholds"):
                st.markdown("### 🎯 **Umbrales de Calidad**")
                st.markdown("""
                **Gradient Boosting:**
                - Premium: ≥ 82.5 puntos
                - Estándar: < 82.5 puntos
                
                **Random Forest:**
                - Premium+ (Excelente): ≥ 85 puntos
                - Premium (Muy Bueno): ≥ 80 puntos
                - Estándar+ (Bueno): ≥ 75 puntos
                - Estándar (Regular): < 75 puntos
                
                **Best Model (SVR):**
                - Premium: ≥ 82.5 puntos (mediana del mercado)
                - Estándar: < 82.5 puntos
                """)
        
        with col2:
            if st.button("🔍 Factores Clave", key="factors"):
                st.markdown("### 🔍 **Factores Clave para Premium**")
                st.markdown("""
                **1. Atributos Sensoriales (Impacto Alto):**
                - Aroma ≥ 8.0/10
                - Sabor ≥ 8.2/10
                - Posgusto ≥ 8.1/10
                - Balance ≥ 8.0/10
                
                **2. Procesamiento (Impacto Alto):**
                - Honey/Natural vs Washed
                - Humedad óptima: 10-12%
                
                **3. Origen (Impacto Medio):**
                - Altitud ≥ 1400m (óptimo: 1600-2000m)
                - Países premium: Etiopía, Kenya, Colombia
                
                **4. Defectos (Impacto Crítico):**
                - Category.One.Defects = 0 para premium
                - Category.Two.Defects ≤ 1 para premium
                """)
        
        with col3:
            if st.button("📈 Estrategias", key="strategies"):
                st.markdown("### 📈 **Estrategias de Mejora**")
                st.markdown("""
                **Control Sensorial (+3-5 puntos):**
                - Mejorar balance de acidez
                - Extender tiempo de extracción
                
                **Mejora de Proceso (+2-3 puntos):**
                - Implementar procesos Honey/Natural
                - Controlar fermentación
                
                **Optimización de Origen (+1-2 puntos):**
                - Buscar fincas ≥ 1400m
                - Preferir variedades premium
                
                **Reducción de Defectos (+1-3 puntos):**
                - Control de calidad riguroso
                - Selección manual de granos
                """)
        
        with col4:
            if st.button("💡 Casos Prácticos", key="cases"):
                st.markdown("### 💡 **Casos de Uso Prácticos**")
                st.markdown("""
                **Productor:**
                - Lote actual: 78 puntos (Estándar)
                - Con mejoras: 84 puntos (Premium)
                
                **Tostador:**
                - Criterios de compra para café premium
                - Validación de calidad objetiva
                
                **Catador:**
                - Evaluación estandarizada
                - Validación cruzada entre modelos
                """)
        
        # Mostrar guía completa
        if st.button("📖 Ver Guía Completa", key="full_guide"):
            st.markdown(guide_content)
            
    except FileNotFoundError:
        st.error("❌ Guía no encontrada. Asegúrate que el archivo 'pages/Guia de calidad de cafes.md' exista.")
    except Exception as e:
        st.error(f"❌ Error al cargar la guía: {e}")

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
