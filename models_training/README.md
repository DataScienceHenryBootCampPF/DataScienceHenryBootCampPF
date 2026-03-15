# 🏆 Coffee Quality Model Training

Módulo completo para el entrenamiento de modelos de Machine Learning destinados a predecir la calidad técnica del café basándose en sus características físicas y químicas.

## 📋 Objetivo

Desarrollar modelos predictivos que puedan estimar el `Total.Cup.Points` (puntaje total de calidad) de un café antes de que un catador humano lo evalúe formalmente, utilizando características como:
- Atributos sensoriales (Aroma, Sabor, Acidez, etc.)
- Variables geográficas (País, Región, Altitud)
- Características de producción (Especie, Variedad, Método de procesamiento)
- Métricas físicas (Humedad, Defectos, Color)

## 🗂️ Estructura del Módulo

```
models_training/
├── preprocessing.py      # Clase para preprocesamiento de datos
├── model_training.py     # Clase para entrenamiento y evaluación de modelos
├── main_training.py      # Script principal para ejecutar el entrenamiento
├── README.md            # Este archivo
└── models/              # Directorio para guardar modelos entrenados
    ├── best_coffee_quality_model.pkl
    ├── *.pkl            # Todos los modelos entrenados
    ├── model_results.csv
    ├── feature_importance.csv
    ├── predictions_report.csv
    └── model_comparison.png
```

## 🚀 Uso Rápido

### 1. Ejecutar Entrenamiento Completo

```bash
cd models_training
python main_training.py
```

### 2. Uso Programático

```python
from model_training import CoffeeQualityModelTrainer
import pandas as pd

# Cargar datos
df = pd.read_csv('../data/processed/coffee_data_cleaned_final.csv')

# Inicializar entrenador
trainer = CoffeeQualityModelTrainer()

# Ejecutar entrenamiento completo
best_model, results = trainer.run_complete_training(df)
```

## 🔧 Componentes Principales

### `CoffeeDataPreprocessor`

Maneja el preprocesamiento completo de datos:

- **Identificación automática** de variables categóricas y numéricas
- **Codificación inteligente**: OneHot para baja cardinalidad, LabelEncoder para alta
- **Escalado estándar** para variables numéricas
- **Manejo de valores nulos** y datos faltantes
- **Análisis de balance de especies** (Arabica vs Robusta)

### `CoffeeQualityModelTrainer`

Orquesta el entrenamiento y evaluación de modelos:

- **Múltiples algoritmos**: LinearRegression, Ridge, Lasso, DecisionTree, RandomForest, GradientBoosting, SVR
- **Optimización de hiperparámetros** con GridSearchCV
- **Validación cruzada** para evaluación robusta
- **Selección automática** del mejor modelo
- **Generación de reportes** y visualizaciones

## 📊 Modelos Implementados

### Modelos Base
1. **LinearRegression** - Regresión lineal clásica
2. **Ridge** - Regresión Ridge con regularización L2
3. **Lasso** - Regresión Lasso con regularización L1
4. **DecisionTree** - Árbol de decisión individual
5. **RandomForest** - Ensemble de árboles aleatorios
6. **GradientBoosting** - Gradient Boosting Machines
7. **SVR** - Support Vector Regression

### Optimización de Hiperparámetros

Cada modelo se optimiza automáticamente con GridSearchCV:

- **Ridge**: alpha [0.1, 1.0, 10.0, 100.0]
- **Lasso**: alpha [0.001, 0.01, 0.1, 1.0]
- **RandomForest**: n_estimators, max_depth, min_samples_split
- **GradientBoosting**: n_estimators, learning_rate, max_depth
- **SVR**: C, gamma

## 📈 Métricas de Evaluación

Los modelos se evalúan con múltiples métricas:

- **RMSE** (Root Mean Squared Error) - Error cuadrático medio
- **MAE** (Mean Absolute Error) - Error absoluto medio
- **R²** (Coeficiente de determinación) - Bondad de ajuste
- **Validación Cruzada** - Estabilidad del modelo

## 🎯 Características Especiales

### Manejo de Desbalance de Clases

- **Análisis automático** de la distribución de especies
- **Estrategias de muestreo** si es necesario
- **Evaluación diferenciada** por especie

### Análisis de Overfitting

- **Comparación Train vs Test** para detectar sobreajuste
- **Validación cruzada** para asegurar generalización
- **Visualizaciones** de rendimiento

### Importancia de Features

- **Extracción automática** de importancia de variables
- **Top 10 features** más influyentes
- **Análisis por modelo** para comparación

## 📁 Archivos de Salida

### Modelos Guardados
- `best_coffee_quality_model.pkl` - Mejor modelo seleccionado
- `{nombre}_model.pkl` - Todos los modelos entrenados

### Reportes
- `model_results.csv` - Tabla comparativa de todos los modelos
- `feature_importance.csv` - Importancia de variables por modelo
- `predictions_report.csv` - Análisis detallado de predicciones

### Visualizaciones
- `model_comparison.png` - Gráficos comparativos de rendimiento

## 🔍 Resultados Esperados

### Métricas Típicas
- **RMSE**: ~1.5-2.5 puntos (en escala 0-100)
- **R²**: ~0.7-0.85 (70-85% de varianza explicada)
- **MAE**: ~1.0-1.8 puntos

### Features Más Importantes
- **Flavor** - Sabor (generalmente la más importante)
- **Aftertaste** - Retrogusto
- **Balance** - Equilibrio general
- **Acidity** - Acidez
- **Body** - Cuerpo/sensación en boca

## 🚨 Consideraciones Importantes

### Desbalance de Especies
- **Arabica**: ~98% de las muestras
- **Robusta**: ~2% de las muestras
- **Impacto**: Los modelos pueden sesgarse hacia características de Arabica

### Preprocesamiento Clave
- **Imputación de valores nulos** necesaria
- **Estandarización de unidades** (altitud en metros)
- **Codificación adecuada** de variables categóricas

### Validación Robusta
- **Cross-validation** esencial para evitar overfitting
- **Múltiples métricas** para evaluación completa
- **Análisis de errores** para mejorar modelo

## 🔄 Flujo de Trabajo

1. **Carga de datos** procesados y limpios
2. **Análisis exploratorio** rápido de calidad
3. **Preprocesamiento** automático de features
4. **Entrenamiento** de múltiples modelos base
5. **Optimización** de hiperparámetros de mejores modelos
6. **Selección** del mejor modelo basado en múltiples criterios
7. **Evaluación** detallada y generación de reportes
8. **Guardado** de modelos y resultados

## 🎯 Aplicación Práctica

El modelo entrenado puede usarse para:

- **Pre-selección** de cafés premium
- **Estimación rápida** de calidad sin catador
- **Identificación** de características clave para mejorar calidad
- **Segmentación** de cafés por rangos de calidad
- **Optimización** de procesos de producción

## 📚 Requisitos

```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

---

**Nota**: Este módulo está diseñado para integrarse perfectamente con el flujo de trabajo existente del proyecto, utilizando los datos ya procesados del pipeline de limpieza.
