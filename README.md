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
