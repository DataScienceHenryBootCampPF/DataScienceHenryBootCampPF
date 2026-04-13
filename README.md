<<<<<<< HEAD
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
=======
# ☕ Coffee Quality Prediction & Recommendation System

## 📋 Descripción del Proyecto

Este proyecto analiza el dataset del Coffee Quality Institute para predecir la calidad del café y desarrollar un sistema de recomendación personalizado. El objetivo es desentrañar los factores físicos (altitud, país, especie) y sensoriales (aroma, cuerpo, acidez) que determinan el Total Cup Points para optimizar la selección y comercialización de café de especialidad.

## 🏗️ Estructura del Proyecto

```
DataScienceHenryBootCampPF/
├── data/                       # Datos del proyecto
│   ├── raw/                    # Dataset original
│   └── processed/              # Datos limpios y procesados
├── notebooks/                  # Jupyter notebooks del análisis
│   ├── 00_preprocessing.ipynb  # Limpieza y preprocesamiento
│   ├── 01_EDA.ipynb           # Análisis Exploratorio de Datos
│   ├── 02_model_analysis.ipynb # Modelos de predicción
│   └── 03_clusterizacion.ipynb # Sistema de recomendación
├── src/                       # Código fuente
│   ├── cleaning.py            # Funciones de limpieza
│   ├── predictor/             # Módulo de predicción
│   └── recommender/           # Módulo de recomendación
├── models/                    # Modelos entrenados
│   ├── prediction/            # Modelos de predicción
│   └── recommender/           # Modelos de clustering
├── reports/                   # Reportes y visualizaciones
│   └── EDA/                   # Análisis exploratorio
│       ├── EDA_REPORT.md      # Reporte completo del EDA
│       └── images/            # Gráficos del análisis
├── metrics/                   # Métricas de evaluación
├── demo_prediction.py         # Demo de predicción
├── demo_recommender.py        # Demo de recomendación
└── requirements.txt           # Dependencias del proyecto
```

## 🚀 Configuración del Entorno

1. **Clonar el repositorio**
>>>>>>> development
```bash
git clone <repository-url>
cd DataScienceHenryBootCampPF
```

<<<<<<< HEAD
2. **Crear entorno virtual**:
=======
2. **Crear entorno virtual**
>>>>>>> development
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows
```

<<<<<<< HEAD
3. **Instalar dependencias**:
=======
3. **Instalar dependencias**
>>>>>>> development
```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
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
=======
## 📊 Análisis Exploratorio de Datos (EDA)

### 🎯 Objetivo Principal
Analizar 1,339 registros de evaluaciones técnicas de café de todo el mundo para identificar patrones que determinen la calidad del grano.

### 🔍 Hallazgos Clave

#### 1. **Distribución de la Calidad**
El dataset presenta un fuerte sesgo hacia la alta calidad, con media de 82.16 puntos y mediana de 82.50. El 75% de los cafés superan los 81 puntos, indicando un mercado de especialidad muy competitivo.

![Distribución Puntaje Total](reports/EDA/images/Dist._PuntajeTotal.png)

#### 2. **Geografía y Excelencia**
Etiopía lidera con el puntaje promedio más alto (85.48) y el máximo histórico (90.58), consolidándose como el benchmark mundial de calidad.

![Puntaje Promedio por País](reports/EDA/images/PuntajeProm_Pais.png)

#### 3. **El Mito de la Altitud**
Contrario a la creencia popular, la correlación entre altitud y calidad es débil (r = 0.20). La calidad depende más del procesamiento humano y la variedad botánica que de los metros sobre el nivel del mar.

![Altura vs Calidad](reports/EDA/images/Altura_vs_Calidad.png)

#### 4. **Métodos de Procesamiento**
El método Washed es el más común, pero procesos artesanales como Honey logran picos de calidad superiores, sugiriendo un mercado creciente para cafés diferenciados.

![Método de Procesamiento vs Calidad](reports/EDA/images/Metodo_Procesamiento_Calidad.png)

#### 5. **Tendencia de Exigencia**
Se observa una caída en puntajes máximos desde 2015-2018, reflejando mayor rigurosidad técnica de los catadores certificados más que una disminución real de la calidad.

![Evolución Puntaje Máximo](reports/EDA/images/Evolución_PuntajeMax.png)

## 🤖 Modelos de Machine Learning

### 📈 Modelo de Predicción de Calidad
- **Algoritmos evaluados**: XGBoost, CatBoost, Random Forest, SVM
- **Target**: Total Cup Points (0-100)
- **Features**: Altitud, país, especie, atributos sensoriales
- **Métricas**: RMSE, MAE, R²

### 🎯 Sistema de Recomendación
- **Técnica**: Clustering K-Means
- **Segmentación**: Perfiles de café basados en características sensoriales
- **Aplicación**: Recomendación personalizada según preferencias del consumidor

## 🛠️ Pipeline de Preprocesamiento

### 1. **Limpieza de Datos**
- Eliminación de columnas con >79% nulos (Lot.Number)
- Remoción de variables sin valor predictivo (IDs, contactos)

### 2. **Tratamiento de Altitud**
- Unificación de unidades (pies → metros)
- Eliminación de outliers espaciales (>4000m)
- Cálculo de altitud media normalizada

### 3. **Imputación Jerárquica**
- Moda para variables categóricas (basado en País + Especie)
- Mediana para variables numéricas (respetando biología de la planta)

### 4. **Feature Engineering**
- Creación de Nombre_Comercial para identidad de marketing
- Eliminación de variables sin varianza (Sweetness, Clean.Cup, Uniformity)

## 🎮 Demostración Interactiva

### Predicción de Calidad
```bash
python demo_prediction.py
```
Permite evaluar diferentes modelos y predecir la calidad de nuevos lotes de café.

### Sistema de Recomendación
```bash
python demo_recommender.py
```
Ofrece recomendaciones personalizadas basadas en perfiles de preferencia.

## 📋 Diccionario de Datos

### Identidad y Trazabilidad
- **Species**: Especie botánica (Arabica/Robusta)
- **Country.of.Origin**: País de procedencia
- **Region**: Zona geográfica específica
- **Altitude**: Rango de altitud de cultivo

### Atributos Sensoriales (0-10 puntos)
- **Aroma**: Fragancia del café
- **Flavor**: Intensidad y calidad del sabor
- **Aftertaste**: Calidad del sabor remanente
- **Acidity**: Nivel de brillantez
- **Body**: Textura y sensación
- **Balance**: Equilibrio general
- **Total.Cup.Points**: Puntaje final (Target)

### Atributos Físicos
- **Moisture**: Grado de humedad
- **Color**: Tono visual del grano
- **Category.One/Two.Defects**: Conteo de defectos

## 🏆 Resultados Principales

1. **Predicción de Calidad**: Modelos XGBoost y CatBoost logran RMSE < 2.5 puntos
2. **Segmentación de Mercado**: 3 clusters bien definidos con perfiles sensoriales distintos
3. **Insights de Negocio**: La altitud no es el factor determinante; el procesamiento humano es clave
4. **Recomendación Personalizada**: Sistema capaz de sugerir cafés según preferencias individuales

## 🔮 Aplicaciones Prácticas

- **Para Productores**: Optimización de procesos para mejorar calidad
- **Para Tostadores**: Selección informada de lotes según perfil deseado
- **Para Consumidores**: Descubrimiento de cafés alineados a gustos personales
- **Para Comerciantes**: Precios basados en calidad objetiva y preferencias de mercado

## 📚 Tecnologías Utilizadas

- **Python 3.x**
- **Pandas & NumPy**: Manipulación de datos
- **Scikit-learn**: Machine Learning
- **XGBoost & CatBoost**: Gradient Boosting
- **Matplotlib & Seaborn**: Visualización
- **Jupyter**: Análisis interactivo

## 👥 Contribuciones

Este proyecto fue desarrollado como parte del Bootcamp de Data Science de Henry, aplicando metodologías de ciencia de datos para resolver un problema real del mercado del café de especialidad.

---

**📧 Contacto**: Para más información sobre este proyecto, consulte los notebooks en la carpeta `notebooks/` o ejecute las demos interactivas.
>>>>>>> development
