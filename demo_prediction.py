"""
Demo de Predicción de Calidad de Café
=====================================

Este script demuestra cómo usar el mejor modelo entrenado
para predecir la calidad del café basándose en características específicas.

Author: Data Science Henry Bootcamp
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

class CoffeeQualityPredictor:
    """
    Clase para predecir la calidad del café usando el modelo entrenado.
    """
    
    def __init__(self):
        """
        Inicializa el predictor cargando el modelo y preprocesador.
        """
        # Rutas a los archivos del modelo
        model_path = "models/prediction/best_model.pkl"
        preprocessor_path = "models/prediction/preprocessor.pkl"
        metadata_path = "models/prediction/training_metadata.pkl"
        
        try:
            # Cargar modelo
            self.model = joblib.load(model_path)
            print(f"✅ Modelo cargado: {type(self.model).__name__}")
            
            # Cargar preprocesador
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"✅ Preprocesador cargado")
            
            # Cargar metadatos
            self.metadata = joblib.load(metadata_path)
            print(f"✅ Metadatos cargados")
            
            # Información del modelo
            print(f"\n📊 INFORMACIÓN DEL MODELO:")
            print(f"   - Mejor modelo: {self.metadata['best_model_name']}")
            print(f"   - RMSE: {self.metadata['best_rmse']:.3f}")
            print(f"   - R²: {self.metadata['best_r2']:.3f}")
            print(f"   - Features: {len(self.metadata['feature_names'])}")
            
        except FileNotFoundError as e:
            print(f"❌ Error: No se encontraron los archivos del modelo: {e}")
            print("💡 Asegúrate de haber ejecutado primero el pipeline de entrenamiento")
            raise
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            raise
    
    def preprocess_input(self, input_data):
        """
        Preprocesa los datos de entrada para el modelo.
        
        Args:
            input_data (dict): Diccionario con características del café
            
        Returns:
            np.array: Datos preprocesados para el modelo
        """
        # Convertir a DataFrame
        df = pd.DataFrame([input_data])
        
        # Aplicar el mismo preprocesamiento que en entrenamiento
        processed_data = self.preprocessor.transform(df)
        
        return processed_data
    
    def predict_quality(self, input_data):
        """
        Predice la calidad del café.
        
        Args:
            input_data (dict): Diccionario con características del café
            
        Returns:
            dict: Resultados de la predicción
        """
        try:
            # Preprocesar datos
            processed_data = self.preprocess_input(input_data)
            
            # Hacer predicción
            prediction = self.model.predict(processed_data)[0]
            
            # Calcular intervalo de confianza (basado en RMSE del modelo)
            rmse = self.metadata['best_rmse']
            lower_bound = max(0, prediction - 1.96 * rmse)  # 95% CI
            upper_bound = min(100, prediction + 1.96 * rmse)  # 95% CI
            
            # Determinar categoría de calidad
            if prediction >= 85:
                quality_category = "Excelente"
                color = "🟢"
            elif prediction >= 80:
                quality_category = "Muy Bueno"
                color = "🔵"
            elif prediction >= 75:
                quality_category = "Bueno"
                color = "🟡"
            else:
                quality_category = "Regular"
                color = "🟠"
            
            results = {
                'predicted_score': round(prediction, 2),
                'quality_category': quality_category,
                'confidence_interval': {
                    'lower': round(lower_bound, 2),
                    'upper': round(upper_bound, 2)
                },
                'model_rmse': rmse,
                'accuracy_estimate': f"{(1 - rmse/100)*100:.1f}%"
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error en la predicción: {e}")
            return None
    
    def print_prediction_result(self, input_data, results):
        """
        Imprime los resultados de la predicción de forma formateada.
        
        Args:
            input_data (dict): Datos de entrada
            results (dict): Resultados de la predicción
        """
        if results is None:
            return
        
        print("\n" + "="*60)
        print("🎯 RESULTADO DE LA PREDICCIÓN")
        print("="*60)
        
        print(f"📊 Puntuación Predicha: {results['predicted_score']}/100")
        print(f"🏆 Categoría de Calidad: {results['quality_category']}")
        print(f"📈 Intervalo de Confianza (95%): {results['confidence_interval']['lower']} - {results['confidence_interval']['upper']}")
        print(f"🎯 Precisión Estimada: {results['accuracy_estimate']}")
        print(f"⚠️ Error del Modelo: ±{results['model_rmse']:.2f} puntos")
        
        print(f"\n📋 Características Evaluadas:")
        for key, value in input_data.items():
            print(f"   - {key}: {value}")
        
        print(f"\n💡 Interpretación:")
        if results['predicted_score'] >= 85:
            print(f"   ✅ Café de especialidad, excelente para el mercado premium")
        elif results['predicted_score'] >= 80:
            print(f"   ✅ Café de alta calidad, muy bueno para el mercado especial")
        elif results['predicted_score'] >= 75:
            print(f"   ✅ Café de buena calidad, adecuado para consumo general")
        else:
            print(f"   ⚠️ Café de calidad regular, podría necesitar mejoras")

def create_sample_coffees():
    """
    Crea ejemplos de café para demostración.
    
    Returns:
        list: Lista de diccionarios con características de café
    """
    sample_coffees = [
        {
            "name": "Café Especial Etiopía",
            "description": "Café de altura, con notas florales y afrutadas",
            "data": {
                'Species': 'Arabica',
                'Number.of.Bags': 300,
                'Aroma': 8.5,
                'Flavor': 8.7,
                'Aftertaste': 8.6,
                'Acidity': 8.8,
                'Body': 8.4,
                'Balance': 8.5,
                'Cupper.Points': 8.6,
                'Moisture': 0.11,
                'Category.One.Defects': 0,
                'Category.Two.Defects': 2,
                'altitude_mean_meters': 1800,
                'Country.of.Origin': 'Ethiopia',
                'Region': 'Yirgacheffe',
                'Variety': 'Heirloom',
                'Processing.Method': 'Washed / Wet',
                'Color': 'Green',
                'Owner': 'Ethiopian Coffee Exporter',
                # Features adicionales creadas por feature engineering
                'altitude_category': 'Media-Alta',
                'altitude_std': 0.5,
                'sensory_avg': 8.6,
                'sensory_std': 0.15,
                'best_sensory': 8.8,
                'total_defects': 2,
                'no_defects': 0,
                'moisture_category': 'Óptima',
                'processing_simple': 'Washed'
            }
        },
        {
            "name": "Café Colombiano Premium",
            "description": "Café balanceado con notas de chocolate y nuez",
            "data": {
                'Species': 'Arabica',
                'Number.of.Bags': 250,
                'Aroma': 8.2,
                'Flavor': 8.4,
                'Aftertaste': 8.3,
                'Acidity': 8.1,
                'Body': 8.5,
                'Balance': 8.3,
                'Cupper.Points': 8.4,
                'Moisture': 0.12,
                'Category.One.Defects': 0,
                'Category.Two.Defects': 1,
                'altitude_mean_meters': 1600,
                'Country.of.Origin': 'Colombia',
                'Region': 'Huila',
                'Variety': 'Caturra',
                'Processing.Method': 'Washed / Wet',
                'Color': 'Green',
                'Owner': 'Colombian Coffee Farm',
                # Features adicionales
                'altitude_category': 'Media',
                'altitude_std': 0.0,
                'sensory_avg': 8.3,
                'sensory_std': 0.12,
                'best_sensory': 8.5,
                'total_defects': 1,
                'no_defects': 0,
                'moisture_category': 'Óptima',
                'processing_simple': 'Washed'
            }
        },
        {
            "name": "Café Robusta Comercial",
            "description": "Café robusto con cuerpo fuerte y amargor característico",
            "data": {
                'Species': 'Robusta',
                'Number.of.Bags': 500,
                'Aroma': 6.5,
                'Flavor': 6.8,
                'Aftertaste': 6.7,
                'Acidity': 6.2,
                'Body': 7.5,
                'Balance': 6.6,
                'Cupper.Points': 6.8,
                'Moisture': 0.13,
                'Category.One.Defects': 1,
                'Category.Two.Defects': 5,
                'altitude_mean_meters': 800,
                'Country.of.Origin': 'Vietnam',
                'Region': 'Central Highlands',
                'Variety': 'Robusta',
                'Processing.Method': 'Natural / Dry',
                'Color': 'Blue-Green',
                'Owner': 'Vietnam Coffee Corporation',
                # Features adicionales
                'altitude_category': 'Baja',
                'altitude_std': -1.5,
                'sensory_avg': 6.7,
                'sensory_std': 0.45,
                'best_sensory': 7.5,
                'total_defects': 6,
                'no_defects': 0,
                'moisture_category': 'Aceptable',
                'processing_simple': 'Natural'
            }
        },
        {
            "name": "Café Kenia AA",
            "description": "Café de altura con acidez brillante y notas cítricas",
            "data": {
                'Species': 'Arabica',
                'Number.of.Bags': 150,
                'Aroma': 8.8,
                'Flavor': 8.9,
                'Aftertaste': 8.7,
                'Acidity': 9.0,
                'Body': 8.3,
                'Balance': 8.6,
                'Cupper.Points': 8.8,
                'Moisture': 0.10,
                'Category.One.Defects': 0,
                'Category.Two.Defects': 0,
                'altitude_mean_meters': 2000,
                'Country.of.Origin': 'Kenya',
                'Region': 'Nyeri',
                'Variety': 'SL28',
                'Processing.Method': 'Washed / Wet',
                'Color': 'Green',
                'Owner': 'Kenya Coffee Estate',
                # Features adicionales
                'altitude_category': 'Alta',
                'altitude_std': 1.0,
                'sensory_avg': 8.7,
                'sensory_std': 0.25,
                'best_sensory': 9.0,
                'total_defects': 0,
                'no_defects': 1,
                'moisture_category': 'Óptima',
                'processing_simple': 'Washed'
            }
        }
    ]
    
    return sample_coffees

def interactive_prediction():
    """
    Función para predicción interactiva por consola.
    """
    print("\n🎮 MODO INTERACTIVO DE PREDICCIÓN")
    print("="*50)
    print("Ingresa las características del café para predecir su calidad.")
    print("Presiona Enter para usar valores por defecto.\n")
    
    input_data = {}
    
    # Características principales
    print("🌱 Características Básicas:")
    input_data['Species'] = input("Especie (Arabica/Robusta) [Arabica]: ") or "Arabica"
    input_data['Aroma'] = float(input("Aroma (0-10) [8.0]: ") or "8.0")
    input_data['Flavor'] = float(input("Sabor (0-10) [8.2]: ") or "8.2")
    input_data['Aftertaste'] = float(input("Posgusto (0-10) [8.1]: ") or "8.1")
    input_data['Acidity'] = float(input("Acidez (0-10) [8.0]: ") or "8.0")
    input_data['Body'] = float(input("Cuerpo (0-10) [8.3]: ") or "8.3")
    input_data['Balance'] = float(input("Balance (0-10) [8.2]: ") or "8.2")
    
    print("\n📍 Origen:")
    input_data['Country.of.Origin'] = input("País de origen [Colombia]: ") or "Colombia"
    input_data['Region'] = input("Región [Huila]: ") or "Huila"
    input_data['altitude_mean_meters'] = float(input("Altitud (metros) [1500]: ") or "1500")
    
    print("\n🔧 Características Técnicas:")
    input_data['Moisture'] = float(input("Humedad (%) [0.12]: ") or "0.12")
    input_data['Category.One.Defects'] = int(input("Defectos Categoría 1 [0]: ") or "0")
    input_data['Category.Two.Defects'] = int(input("Defectos Categoría 2 [1]: ") or "1")
    
    # Valores por defecto para campos menos importantes
    default_values = {
        'Number.of.Bags': 300,
        'Cupper.Points': 8.0,
        'Variety': 'Unknown',
        'Processing.Method': 'Washed / Wet',
        'Color': 'Green',
        'Owner': 'Unknown Farm'
    }
    
    for key, value in default_values.items():
        if key not in input_data:
            input_data[key] = value
    
    return input_data

def main():
    """
    Función principal del demo.
    """
    print("🚀 DEMO DE PREDICCIÓN DE CALIDAD DE CAFÉ")
    print("="*60)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Objetivo: Predecir la calidad del café usando ML")
    
    try:
        # Inicializar predictor
        predictor = CoffeeQualityPredictor()
        
        # Menú de opciones
        print("\n📋 MENÚ DE OPCIONES:")
        print("1. Probar ejemplos predefinidos")
        print("2. Predicción interactiva")
        print("3. Salir")
        
        option = input("\nSelecciona una opción [1]: ") or "1"
        
        if option == "1":
            # Probar ejemplos predefinidos
            sample_coffees = create_sample_coffees()
            
            print(f"\n🧪 PROBANDO {len(sample_coffees)} EJEMPLOS DE CAFÉ")
            print("="*60)
            
            for i, coffee in enumerate(sample_coffees, 1):
                print(f"\n🔸 EJEMPLO {i}: {coffee['name']}")
                print(f"📝 {coffee['description']}")
                
                # Predecir
                results = predictor.predict_quality(coffee['data'])
                
                # Mostrar resultados
                predictor.print_prediction_result(coffee['data'], results)
                
                if i < len(sample_coffees):
                    input("\nPresiona Enter para continuar al siguiente ejemplo...")
        
        elif option == "2":
            # Predicción interactiva
            input_data = interactive_prediction()
            
            print(f"\n🔮 REALIZANDO PREDICCIÓN...")
            results = predictor.predict_quality(input_data)
            
            predictor.print_prediction_result(input_data, results)
        
        elif option == "3":
            print("👋 ¡Hasta luego!")
            return
        
        else:
            print("❌ Opción no válida")
        
        print(f"\n✅ Demo completado exitosamente!")
        print(f"💡 Para más análisis, revisa el notebook: notebooks/model_analysis.ipynb")
        
    except Exception as e:
        print(f"❌ Error en el demo: {e}")
        print("💡 Asegúrate de haber ejecutado primero el pipeline completo de entrenamiento")

if __name__ == "__main__":
    main()
