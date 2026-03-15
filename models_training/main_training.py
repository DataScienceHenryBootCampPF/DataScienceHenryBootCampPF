import pandas as pd
import numpy as np
import sys
import os
from model_training import CoffeeQualityModelTrainer
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Función principal para ejecutar el entrenamiento de modelos.
    """
    print("🚀 Iniciando Sistema de Entrenamiento de Modelos de Calidad de Café")
    print("=" * 70)
    
    # 1. Cargar datos procesados
    print("\n📁 Cargando dataset procesado...")
    
    # Ruta al archivo de datos procesados
    data_path = "./data/processed/coffee_data_cleaned_final.csv"
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Dataset cargado exitosamente: {df.shape}")
        print(f"   - Filas: {df.shape[0]}")
        print(f"   - Columnas: {df.shape[1]}")
        print(f"   - Variable objetivo: Total.Cup.Points")
        
        # Verificar que la variable objetivo existe
        if 'Total.Cup.Points' not in df.columns:
            raise ValueError("La columna 'Total.Cup.Points' no se encuentra en el dataset")
            
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {data_path}")
        print("   Asegúrate de haber ejecutado primero el notebook de limpieza")
        return
    except Exception as e:
        print(f"❌ Error al cargar los datos: {e}")
        return
    
    # 2. Análisis preliminar de los datos
    print("\n📊 Análisis preliminar de datos:")
    
    # Estadísticas básicas de la variable objetivo
    target_stats = df['Total.Cup.Points'].describe()
    print(f"   - Puntaje mínimo: {target_stats['min']:.2f}")
    print(f"   - Puntaje máximo: {target_stats['max']:.2f}")
    print(f"   - Puntaje promedio: {target_stats['mean']:.2f}")
    print(f"   - Desviación estándar: {target_stats['std']:.2f}")
    
    # Análisis de especies
    if 'Species' in df.columns:
        species_counts = df['Species'].value_counts()
        print(f"\n🌱 Distribución de especies:")
        for species, count in species_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   - {species}: {count} muestras ({percentage:.1f}%)")
    
    # 3. Verificar calidad de datos
    print("\n🔍 Verificación de calidad de datos:")
    
    # Verificar valores nulos en variables importantes
    null_counts = df.isnull().sum()
    high_null_columns = null_counts[null_counts > 0]
    
    if len(high_null_columns) > 0:
        print("⚠️  Columnas con valores nulos:")
        for col, count in high_null_columns.items():
            print(f"   - {col}: {count} nulos ({(count/len(df))*100:.1f}%)")
    else:
        print("✅ No se detectaron valores nulos")
    
    # Verificar valores atípicos en variable objetivo
    q1 = df['Total.Cup.Points'].quantile(0.25)
    q3 = df['Total.Cup.Points'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df['Total.Cup.Points'] < lower_bound) | (df['Total.Cup.Points'] > upper_bound)]
    print(f"   - Outliers en puntaje: {len(outliers)} muestras ({(len(outliers)/len(df))*100:.1f}%)")
    
    # 4. Inicializar y ejecutar entrenamiento
    print("\n🎯 Iniciando entrenamiento de modelos...")
    
    try:
        # Crear directorio para modelos si no existe
        models_dir = "./models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Inicializar entrenador
        trainer = CoffeeQualityModelTrainer(model_save_path=models_dir)
        
        # Ejecutar entrenamiento completo
        best_model, results_df = trainer.run_complete_training(df)
        
        # 5. Resumen final
        print("\n" + "=" * 70)
        print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print("=" * 70)
        
        print(f"\n🥇 MEJOR MODELO: {trainer.best_model_name}")
        best_results = results_df[results_df['Model'] == trainer.best_model_name].iloc[0]
        print(f"   📊 RMSE (Test): {best_results['Test_RMSE']:.3f}")
        print(f"   📊 R² (Test): {best_results['Test_R2']:.3f}")
        
        print(f"\n📈 TOP 3 MODELOS:")
        top_models = results_df.nsmallest(3, 'Test_RMSE')[['Model', 'Test_RMSE', 'Test_R2']]
        for idx, (model, rmse, r2) in enumerate(top_models.values, 1):
            print(f"   {idx}. {model}: RMSE={rmse:.3f}, R²={r2:.3f}")
        
        print(f"\n💾 ARCHIVOS GUARDADOS:")
        print(f"   - Modelos: ./models/*.pkl")
        print(f"   - Resultados: ./models/model_results.csv")
        print(f"   - Importancia: ./models/feature_importance.csv")
        print(f"   - Reporte: ./models/predictions_report.csv")
        print(f"   - Gráficos: ./models/model_comparison.png")
        
        print(f"\n🔮 USO DEL MODELO:")
        print(f"   Para hacer predicciones, carga el mejor modelo:")
        print(f"   >>> import joblib")
        print(f"   >>> model = joblib.load('./models/best_coffee_quality_model.pkl')")
        print(f"   >>> prediction = model.predict(X_preprocessed)")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        print("   Revisa los datos y la configuración del modelo")
        return
    
    print("\n✨ Proceso finalizado. Los modelos están listos para producción!")

if __name__ == "__main__":
    main()
