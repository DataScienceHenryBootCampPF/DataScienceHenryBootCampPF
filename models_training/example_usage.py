"""
Ejemplo de cómo usar los modelos entrenados para hacer predicciones
con nuevos datos de café.
"""

import pandas as pd
import numpy as np
import joblib
from preprocessing import CoffeeDataPreprocessor

def load_trained_model(model_path='./models/best_coffee_quality_model.pkl'):
    """
    Carga el mejor modelo entrenado.
    """
    try:
        model = joblib.load(model_path)
        print(f"✅ Modelo cargado desde: {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ No se encontró el modelo en: {model_path}")
        print("   Asegúrate de haber ejecutado el entrenamiento primero")
        return None

def prepare_new_data(new_coffee_data):
    """
    Prepara nuevos datos de café para predicción.
    
    Args:
        new_coffee_data: DataFrame con las mismas columnas que los datos de entrenamiento
    
    Returns:
        Datos preprocesados listos para predicción
    """
    # Cargar datos de entrenamiento para ajustar preprocesador
    train_data = pd.read_csv('./data/processed/coffee_data_cleaned_final.csv')
    
    # Inicializar preprocesador
    preprocessor = CoffeeDataPreprocessor()
    
    # Ajustar preprocesador con datos de entrenamiento
    preprocessor.identify_variable_types(train_data)
    preprocessor.fit_label_encoders(train_data)
    preprocessor.create_preprocessor(train_data)
    
    # Fittear el column transformer con datos de entrenamiento
    X_train = train_data[preprocessor.feature_columns].copy()
    X_train = preprocessor.transform_categorical_features(X_train)
    preprocessor.column_transformer.fit(X_train)
    
    # Preparar nuevos datos
    X_new = new_coffee_data[preprocessor.feature_columns].copy()
    X_new = preprocessor.transform_categorical_features(X_new)
    X_new_processed = preprocessor.column_transformer.transform(X_new)
    
    return X_new_processed

def predict_coffee_quality(model, new_coffee_data):
    """
    Realiza predicciones de calidad para nuevos datos de café.
    
    Args:
        model: Modelo entrenado cargado
        new_coffee_data: DataFrame con datos de café a predecir
    
    Returns:
        Predicciones de calidad
    """
    # Preparar datos
    X_processed = prepare_new_data(new_coffee_data)
    
    # Hacer predicciones
    predictions = model.predict(X_processed)
    
    return predictions

def example_prediction():
    """
    Ejemplo completo de cómo hacer predicciones.
    """
    print("🔮 Ejemplo de Predicción de Calidad de Café")
    print("=" * 50)
    
    # 1. Cargar modelo entrenado
    model = load_trained_model()
    if model is None:
        return
    
    # 2. Crear datos de ejemplo (simulando nuevos cafés con valores realistas)
    sample_coffees = pd.DataFrame({
        'Species': ['Arabica', 'Arabica', 'Robusta'],
        'Owner': ['metad plc', 'ethiopia commodity exchange', 'vietnam coffee corp'],
        'Country.of.Origin': ['Colombia', 'Ethiopia', 'Vietnam'],
        'Region': ['Huila', 'Yirgacheffe', 'Central Highlands'],
        'Number.of.Bags': [100, 150, 200],
        'In.Country.Partner': ['Almacafé', 'Ethiopia Commodity Exchange', 'Vietnam Coffee Corp'],
        'Grading.Date': ['2024-01-15', '2024-02-20', '2024-03-10'],
        'Variety': ['Caturra', 'Heirloom', 'Robusta'],
        'Processing.Method': ['Washed / Wet', 'Natural / Dry', 'Semi-Washed / Semi-Pulped'],
        'Aroma': [7.5, 8.0, 6.5],
        'Flavor': [7.8, 8.2, 6.8],
        'Aftertaste': [7.6, 8.1, 6.7],
        'Acidity': [7.7, 8.3, 6.4],
        'Body': [7.4, 7.9, 7.0],
        'Balance': [7.5, 8.0, 6.6],
        'Cupper.Points': [7.6, 8.1, 6.8],
        'Moisture': [0.11, 0.10, 0.12],
        'Category.One.Defects': [0, 0, 1],
        'Color': ['Green', 'Green', 'Blue-Green'],
        'Category.Two.Defects': [2, 1, 3],
        'altitude_mean_meters': [1500, 1800, 800],
        'categoria_altitud': ['Alta (Arabica)', 'Premium (Arabica)', 'Media (Robusta)']
    })
    
    print("\n📊 Datos de café para predicción:")
    print(sample_coffees[['Species', 'Country.of.Origin', 'Variety', 'Processing.Method']].to_string())
    
    # 3. Hacer predicciones
    predictions = predict_coffee_quality(model, sample_coffees)
    
    # 4. Mostrar resultados
    print("\n🎯 Resultados de Predicción:")
    print("-" * 50)
    
    for i, (idx, row) in enumerate(sample_coffees.iterrows()):
        predicted_score = predictions[i]
        quality_category = categorize_quality(predicted_score)
        
        print(f"Café {i+1}: {row['Species']} - {row['Country.of.Origin']}")
        print(f"   📈 Puntaje Predicho: {predicted_score:.2f}")
        print(f"   🏆 Categoría: {quality_category}")
        print(f"   🌱 Variedad: {row['Variety']}")
        print(f"   ⚙️  Procesamiento: {row['Processing.Method']}")
        print()

def categorize_quality(score):
    """
    Categoriza el puntaje de calidad en niveles descriptivos.
    """
    if score >= 90:
        return "Excelente / Premium"
    elif score >= 85:
        return "Muy Bueno"
    elif score >= 80:
        return "Bueno"
    elif score >= 75:
        return "Aceptable"
    else:
        return "Mejorable"

def batch_prediction_example():
    """
    Ejemplo de predicción en lote para múltiples cafés.
    """
    print("\n🔄 Ejemplo de Predicción en Lote")
    print("=" * 40)
    
    # Cargar modelo
    model = load_trained_model()
    if model is None:
        return
    
    # Cargar dataset de prueba (usando una porción de los datos originales)
    try:
        full_data = pd.read_csv('./data/processed/coffee_data_cleaned_final.csv')
        
        # Tomar una muestra para predicción
        sample_data = full_data.sample(10, random_state=42)
        
        # Separar las columnas que no se usan para predicción
        prediction_data = sample_data.drop('Total.Cup.Points', axis=1)
        actual_scores = sample_data['Total.Cup.Points']
        
        # Hacer predicciones
        predictions = predict_coffee_quality(model, prediction_data)
        
        # Comparar predicciones vs realidad
        print("\n📊 Comparación Predicciones vs Realidad:")
        print("-" * 60)
        print(f"{'Café':<6} {'Real':<8} {'Predicho':<10} {'Error':<10} {'Categoría':<15}")
        print("-" * 60)
        
        total_error = 0
        for i, (actual, predicted) in enumerate(zip(actual_scores, predictions)):
            error = abs(actual - predicted)
            total_error += error
            category = categorize_quality(predicted)
            
            print(f"{i+1:<6} {actual:<8.2f} {predicted:<10.2f} {error:<10.2f} {category:<15}")
        
        print("-" * 60)
        print(f"Error Promedio: {total_error/len(predictions):.2f} puntos")
        
    except FileNotFoundError:
        print("❌ No se encontraron los datos de prueba")

if __name__ == "__main__":
    # Ejecutar ejemplo de predicción
    example_prediction()
    
    # Ejecutar ejemplo de predicción en lote
    batch_prediction_example()
