"""
Módulo de Predicción de Calidad de Café
=====================================
- Feature Engineering para preparación de datos
- Entrenamiento de modelos de Machine Learning
- Predicción de calidad de café

Author: Data Science Henry Bootcamp
Version: 1.0.0
"""

import os
import sys

# Agregar el directorio actual al path para imports relativos
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Importar las clases
from feature_engineering import CoffeeFeatureEngineer
from model_training import CoffeeModelTrainer

__all__ = [
    'CoffeeFeatureEngineer',
    'CoffeeModelTrainer'
]

# Ejecutar feature engineering si se ejecuta directamente
if __name__ == "__main__":
    # Obtener el directorio del proyecto
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    data_path = os.path.join(project_root, "data", "processed", "coffee_data_cleaned_final.csv")
    output_dir = os.path.join(project_root, "models", "prediction")
    
    print(f"📁 Ruta corregida de datos: {data_path}")
    print(f"📁 Ruta corregida de salida: {output_dir}")
    
    engineer = CoffeeFeatureEngineer(data_path, output_dir)
    processed_path = engineer.run_full_pipeline()
    
    if processed_path:
        print(f"\n🎉 Feature engineering completado!")
        print(f"📁 Usa el archivo {processed_path} para entrenar modelos")
