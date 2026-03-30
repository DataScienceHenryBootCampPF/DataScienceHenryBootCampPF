# ☕ Coffee Quality Predictor & Recommendation System

Aplicación web interactiva para predecir la calidad del café y encontrar recomendaciones personalizadas utilizando machine learning.

## 🚀 Características

### 🎯 Predicción de Calidad de Café
- **Entrada Manual**: Ingresa las características sensoriales y de origen del café
- **Ejemplos Predefinidos**: Analiza cafés de ejemplo de diferentes regiones
- **Procesamiento por Lotes**: Sube archivos CSV para análisis masivo
- **Visualizaciones Interactivas**: Gráficos y métricas en tiempo real

### 🔍 Sistema de Recomendación Híbrido
- **Perfil Personalizado**: Define tus preferencias sensoriales
- **Búsqueda por Similitud**: Encuentra cafés con perfiles similares
- **Filtros Avanzados**: Filtra por especie, país, región
- **Comparación Visual**: Radar charts comparativos

### 📊 Análisis de Modelos
- **Métricas de Rendimiento**: RMSE, R², precisión
- **Comparación de Modelos**: Visualización del rendimiento de diferentes algoritmos
- **Importancia de Features**: Análisis de características más relevantes

## 🛠️ Instalación

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd DataScienceHenryBootCampPF
```

2. **Crear entorno virtual**:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

## 🎮 Uso

### Iniciar la Aplicación
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Navegación
La aplicación tiene 4 secciones principales:

1. **🎯 Predicción de Calidad**: Predice la puntuación de calidad del café
2. **🔍 Sistema de Recomendación**: Encuentra cafés similares a tus preferencias
3. **📊 Análisis de Modelos**: Explora las métricas y rendimiento de los modelos
4. **ℹ️ Información**: Detalles técnicos y uso del sistema

## 📋 Requisitos del Sistema

### Para Predicción de Calidad
El modelo requiere las siguientes características:

#### Características Sensoriales (0-10)
- **Aroma**: Intensidad y complejidad del aroma
- **Sabor**: Perfil de sabor general
- **Posgusto**: Persistencia y calidad del aftertaste
- **Acidez**: Brillantez y vivacidad
- **Cuerpo**: Peso y textura en boca
- **Balance**: Equilibrio general

#### Características de Origen
- **Especie**: Arabica o Robusta
- **País de Origen**: País donde se produjo el café
- **Altitud**: Altitud en metros sobre el nivel del mar
- **Humedad**: Contenido de humedad (%)
- **Defectos**: Categoría 1 y 2 (cantidad)

### Para Sistema de Recomendación
- Especifica tus preferencias en las características sensoriales
- Opcional: Filtra por especie (Arabica/Robusta)
- Ajusta el umbral de similitud y número de resultados

## 🤖 Modelo Machine Learning

### Algoritmos Evaluados
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression (SVR)
- Decision Tree

### Métricas de Evaluación
- **RMSE**: Root Mean Square Error (error cuadrático medio)
- **R²**: Coeficiente de determinación
- **Precisión Estimada**: Basada en el RMSE del modelo

### Características del Mejor Modelo
- **Modelo Seleccionado**: Automáticamente elige el mejor rendimiento
- **Features**: 25+ características incluyendo sensoriales, de origen y técnicas
- **Preprocesamiento**: Estandarización y encoding de variables categóricas

## 📊 Estructura del Proyecto

```
DataScienceHenryBootCampPF/
├── app.py                          # Aplicación principal de Streamlit
├── requirements.txt                # Dependencias del proyecto
├── demo_prediction.py             # Script de demostración original
├── README.md                       # Este archivo
├── data/                          # Datos de entrenamiento
├── models/                        # Modelos entrenados
│   └── prediction/
│       ├── best_model.pkl         # Mejor modelo predictivo
│       ├── preprocessor.pkl       # Preprocesador de datos
│       ├── training_metadata.pkl  # Metadatos del entrenamiento
│       └── all_models/           # Todos los modelos evaluados
├── src/                          # Código fuente
│   └── predictor/
│       └── recommendation_system.py
└── notebooks/                     # Notebooks de análisis
```

## 🎯 Ejemplos de Uso

### Predicción de Calidad
```python
# Ejemplo de café especial etíope
input_data = {
    'Species': 'Arabica',
    'Aroma': 8.5,
    'Flavor': 8.7,
    'Aftertaste': 8.6,
    'Acidity': 8.8,
    'Body': 8.4,
    'Balance': 8.5,
    'Country.of.Origin': 'Ethiopia',
    'altitude_mean_meters': 1800,
    'Moisture': 0.11,
    'Category.One.Defects': 0,
    'Category.Two.Defects': 2
}
```

### Sistema de Recomendación
```python
# Perfil de preferencias
preferences = {
    'Flavor': 8.5,
    'Aftertaste': 8.3,
    'Aroma': 8.7,
    'Acidity': 8.2,
    'Body': 8.4,
    'Balance': 8.6
}
```

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
# Opcional: Configurar puerto de Streamlit
STREAMLIT_SERVER_PORT=8501

# Opcional: Configurar dirección
STREAMLIT_SERVER_ADDRESS=localhost
```

### Personalización
- Modifica `app.py` para ajustar la interfaz
- Agrega nuevos modelos en `models/prediction/`
- Extiende el sistema de recomendación en `src/predictor/`

## 🐈‍⬛ Troubleshooting

### Problemas Comunes

1. **Error cargando modelos**:
   - Verifica que los archivos `.pkl` existan en `models/prediction/`
   - Asegúrate de haber ejecutado el pipeline de entrenamiento

2. **Error en sistema de recomendación**:
   - Verifica que los datos de café estén disponibles
   - Revisa el formato de los datos en `src/predictor/recommendation_system.py`

3. **Problemas con Streamlit**:
   - Actualiza Streamlit: `pip install --upgrade streamlit`
   - Limpia el caché: `streamlit cache clear`

### Logs y Depuración
- Los errores se muestran en la interfaz de Streamlit
- Revisa la consola para mensajes detallados
- Habilita el modo debug: `streamlit run app.py --logger.level debug`

## 🚀 Despliegue

### Opción 1: Streamlit Cloud
1. Sube el código a GitHub
2. Conecta tu repositorio en [Streamlit Cloud](https://share.streamlit.io/)
3. Configura el archivo `requirements.txt`

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
