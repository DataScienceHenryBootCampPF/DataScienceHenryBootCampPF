# Coffee Quality Prediction & Recommendation System

## Descripción del Proyecto

Este proyecto analiza el dataset del Coffee Quality Institute para predecir la calidad del café y desarrollar un sistema de recomendación personalizado. El objetivo es desentrañar los factores físicos (altitud, país, especie) y sensoriales (aroma, cuerpo, acidez) que determinan el Total Cup Points para optimizar la selección y comercialización de café de especialidad.

## Estructura del Proyecto

```
DataScienceHenryBootCampPF/
    app_main.py                     # Aplicación principal de Streamlit
    demo_prediction.py              # Demo de predicción (consola)
    demo_recommender.py             # Demo de recomendación (consola)
    requirements.txt                # Dependencias del proyecto
    README.md                       # Documentación del proyecto
    .streamlit/                     # Configuración de Streamlit
    and config.toml                # Configuración del servidor y tema
    
    pages/                          # Páginas de la aplicación Streamlit
    and 1_Gradient_Boosting.py     # Modelo Gradient Boosting - Matías Gutierrez
    and 2_Random_Forest.py         # Modelo Random Forest - Felipe Baquero
    and 3_Best_Model.py            # Sistema dinámico - Matías & Felipe
    
    data/                          # Datos del proyecto
    and raw/                       # Dataset original sin procesar
    and processed/                 # Datos limpios y transformados
    and external/                  # Datos adicionales de referencia
    
    models/                        # Modelos entrenados y artefactos
    and prediction/                # Modelos de predicción de calidad
    and and coffee_model_RandomForest.pkl
    and and coffee_model_GradientBoosting.pkl
    and and coffee_model_SVR.pkl
    and and coffee_model_XGBoost.pkl
    and and coffee_model_CatBoost.pkl
    and and coffee_model_Voting_Ensemble.pkl
    and and preprocessor.pkl       # Preprocesador de datos
    and and training_metadata.pkl  # Metadatos del entrenamiento
    and and categorias_validas.json # Categorías válidas para validación
    and and training_results.csv   # Resultados comparativos
    
    and recommender/               # Sistema de recomendación
    and and kmeans_flavor.pkl      # Modelo de clustering
    and and scaler_recommendation.pkl # Escalador para recomendaciones
    
    src/                          # Código fuente modular
    and cleaning.py               # Funciones de limpieza de datos
    and predictor/                # Módulo de predicción
    and and model_training.py     # Entrenamiento de modelos
    and and recommendation_system.py # Sistema de recomendación híbrido
    
    notebooks/                     # Análisis exploratorio y desarrollo
    and 00_preprocessing.ipynb    # Limpieza y preprocesamiento
    and 01_EDA.ipynb             # Análisis Exploratorio de Datos
    and 02_model_analysis.ipynb   # Análisis comparativo de modelos
    and 03_clusterizacion.ipynb  # Sistema de recomendación
    
    metrics/                       # Métricas y evaluación
    and model_comparison.png      # Visualización comparativa
    and performance_metrics.csv   # Métricas detalladas
    
    reports/                       # Reportes generados
    and final_analysis.pdf         # Análisis completo del proyecto
    
    .github/                       # Configuración de GitHub
    and workflows/                # Acciones automatizadas
```

## Configuración del Entorno

1. **Clonar el repositorio**
```bash
git clone <repository-url>
cd DataScienceHenryBootCampPF
```

2. **Crear entorno virtual**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

## Uso de la Aplicación

### Iniciar la Aplicación Localmente
```bash
streamlit run app_main.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Acceso a la Aplicación en la Nube
**Aplicación en producción:** [https://datasciencehenrybootcamppf-vqjj26ywtdnrcvu9dsedwd.streamlit.app](https://datasciencehenrybootcamppf-vqjj26ywtdnrcvu9dsedwd.streamlit.app)

### Navegación
La aplicación tiene 4 secciones principales:

1. **Página Principal**: Bienvenida y navegación general
2. **Gradient Boosting**: Modelo especializado de Matías Gutierrez
3. **Random Forest**: Modelo especializado de Felipe Baquero  
4. **Best Model**: Sistema dinámico con todos los modelos y demos interactivos

## Despliegue

### Opción 1: Streamlit Cloud (Recomendado)

**Aplicación en producción:** [https://datasciencehenrybootcamppf-vqjj26ywtdnrcvu9dsedwd.streamlit.app](https://datasciencehenrybootcamppf-vqjj26ywtdnrcvu9dsedwd.streamlit.app)

#### ¿Cómo funciona Streamlit Cloud?
Streamlit Cloud es la forma más sencilla de desplegar aplicaciones Streamlit. Funciona automáticamente 24/7 sin necesidad de mantener servidores.

#### Pasos para despliegue:
1. **Prepara tu repositorio**:
   - Asegúrate que `requirements.txt` esté completo
   - Verifica que `app_main.py` sea el archivo principal
   - Sube todo a GitHub

2. **Configura en Streamlit Cloud**:
   - Ve a [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Inicia sesión con GitHub
   - Haz clic en "New app"
   - Selecciona tu repositorio y rama
   - Configura `app_main.py` como archivo principal

3. **Obtén tu enlace público**:
   - Streamlit Cloud genera automáticamente un enlace
   - Tu aplicación queda disponible 24/7
   - Escala automáticamente según el tráfico

#### Ventajas:
- **Gratis** para proyectos públicos
- **Automático**: Detecta cambios y redeploya
- **Sin configuración** de servidores
- **Escalable** automáticamente
- **HTTPS incluido** por defecto

#### Características de nuestra implementación:
- **Modelos de Machine Learning**: Random Forest, Gradient Boosting, Best Model
- **Sistema de recomendación**: Búsqueda inteligente de cafés
- **Visualizaciones interactivas**: Gráficos con Plotly
- **Interfaz moderna**: Diseño responsivo con Streamlit
- **Procesamiento en tiempo real**: Predicciones instantáneas

### Opción 2: Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

### Opción 3: Heroku/Railway/Render
- Configura el buildpack para Python
- Establece el comando start: `streamlit run app.py --server.port=$PORT`

## 🤝 Contribuciones

1. Fork del proyecto
2. Crear rama feature: `git checkout -b feature/nueva-funcionalidad`
3. Commit de cambios: `git commit -am 'Agregar nueva funcionalidad'`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Pull Request

## 📄 Licencia

Este proyecto es parte del bootcamp de Data Science de Henry y está disponible para fines educativos.

## 👥 Equipo

**Data Science Henry Bootcamp - Proyecto Final**

Desarrollado por el equipo de Data Science Henry Bootcamp como proyecto integrador de machine learning y desarrollo de aplicaciones.

---

☕ **Disfruta explorando el mundo del café con data science!**
