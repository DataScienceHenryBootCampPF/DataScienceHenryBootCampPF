"""
Feature Engineering para Predicción de Calidad de Café
======================================================

Este módulo se encarga de:
- Cargar y limpiar los datos originales
- Transformar variables para modelos de ML
- Manejar desbalance de clases
- Guardar datos procesados para entrenamiento

Author: Data Science Henry Bootcamp
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

class CoffeeFeatureEngineer:
    """
    Clase para realizar feature engineering en datos de café.
    
    Esta clase prepara los datos crudos para el entrenamiento de modelos
    de Machine Learning, manejando transformaciones y desbalance.
    """
    
    def __init__(self, data_path, output_dir):
        """
        Inicializa el feature engineer.
        
        Args:
            data_path (str): Ruta al archivo CSV de datos limpios
            output_dir (str): Directorio donde guardar los resultados
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.preprocessor = None
        self.feature_names = None
        
        # Asegurar que el directorio de salida exista
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔧 CoffeeFeatureEngineer inicializado")
        print(f"📁 Datos de entrada: {data_path}")
        print(f"📁 Directorio de salida: {output_dir}")
    
    def load_data(self):
        """
        Carga los datos desde el archivo CSV.
        
        Returns:
            bool: True si se cargaron correctamente, False otherwise
        """
        try:
            print(f"\n📁 Cargando datos desde: {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Datos cargados: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo {self.data_path}")
            return False
        except Exception as e:
            print(f"❌ Error al cargar datos: {str(e)}")
            return False
    
    def analyze_data(self):
        """
        Analiza los datos cargados y muestra estadísticas importantes.
        """
        if self.df is None:
            print("❌ No hay datos cargados. Ejecuta load_data() primero.")
            return
        
        print(f"\n📊 ANÁLISIS EXPLORATORIO DE DATOS")
        print(f"="*50)
        
        # Información básica
        print(f"📁 Dimensiones: {self.df.shape}")
        print(f"🎯 Variable objetivo: Total.Cup.Points")
        
        # Análisis de variable objetivo
        target_col = 'Total.Cup.Points'
        if target_col in self.df.columns:
            print(f"\n📈 Variable Objetivo ({target_col}):")
            print(f"   - Mínimo: {self.df[target_col].min():.2f}")
            print(f"   - Máximo: {self.df[target_col].max():.2f}")
            print(f"   - Media: {self.df[target_col].mean():.2f}")
            print(f"   - Desviación: {self.df[target_col].std():.2f}")
        
        # Análisis de especies (desbalance)
        if 'Species' in self.df.columns:
            print(f"\n🌱 Distribución de Especies:")
            species_counts = self.df['Species'].value_counts()
            total_samples = len(self.df)
            
            for species, count in species_counts.items():
                percentage = (count / total_samples) * 100
                print(f"   - {species}: {count} ({percentage:.1f}%)")
            
            # Alerta de desbalance
            if species_counts.iloc[0] / species_counts.iloc[1] > 10:
                print(f"⚠️ ALERTA: Desbalance significativo detectado")
        
        # Valores nulos
        null_counts = self.df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"\n❓ Valores Nulos:")
            for col, nulls in null_counts[null_counts > 0].items():
                print(f"   - {col}: {nulls}")
        else:
            print(f"\n✅ No hay valores nulos")
    
    def clean_data(self):
        """
        Limpia y prepara los datos para el feature engineering.
        
        Returns:
            pd.DataFrame: DataFrame limpio
        """
        if self.df is None:
            print("❌ No hay datos cargados")
            return None
        
        print(f"\n🧹 LIMPIEZA DE DATOS")
        print(f"="*30)
        
        # Copiar para no modificar original
        df_clean = self.df.copy()
        
        # Eliminar columnas que no se usarán
        columns_to_drop = [
            'Bag.Weight',           # Peso del bolso (no relevante para calidad)
            'In.Country.Partner',   # Socio en país (información administrativa)
            'Grading.Date',        # Fecha de calificación (temporal)
            'categoria_altitud'     # Categoría redundante (ya tenemos altitude_mean_meters)
        ]
        
        existing_columns = [col for col in columns_to_drop if col in df_clean.columns]
        
        if existing_columns:
            df_clean = df_clean.drop(columns=existing_columns)
            print(f"🗑️ Columnas eliminadas: {existing_columns}")
        
        # Manejar valores nulos si existen
        null_columns = df_clean.columns[df_clean.isnull().any()].tolist()
        if null_columns:
            print(f"🔧 Tratando valores nulos en: {null_columns}")
            
            # Para numéricas: rellenar con mediana
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in null_columns:
                if col in numeric_cols:
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    print(f"   - {col}: rellenado con mediana {median_val:.2f}")
            
            # Para categóricas: rellenar con moda
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            for col in null_columns:
                if col in categorical_cols:
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    print(f"   - {col}: rellenado con moda '{mode_val}'")
        
        # Verificar que no queden nulos
        remaining_nulls = df_clean.isnull().sum().sum()
        if remaining_nulls == 0:
            print(f"✅ No hay valores nulos después de la limpieza")
        else:
            print(f"⚠️ Quedan {remaining_nulls} valores nulos")
        
        self.df = df_clean
        return df_clean
    
    def create_features(self):
        """
        Crea nuevas features a partir de las existentes.
        
        Returns:
            pd.DataFrame: DataFrame con nuevas features
        """
        if self.df is None:
            print("❌ No hay datos cargados")
            return None
        
        print(f"\n🔧 CREACIÓN DE FEATURES")
        print(f"="*30)
        
        df_features = self.df.copy()
        
        # 1. Features de altitud (si existe)
        if 'altitude_mean_meters' in df_features.columns:
            # Categorías de altitud
            df_features['altitude_category'] = pd.cut(
                df_features['altitude_mean_meters'],
                bins=[0, 1000, 1500, 2000, 3000, float('inf')],
                labels=['Baja', 'Media-Baja', 'Media', 'Media-Alta', 'Alta']
            )
            print(f"✅ Categoría de altitud creada")
            
            # Altitud estandarizada
            altitude_mean = df_features['altitude_mean_meters'].mean()
            altitude_std = df_features['altitude_mean_meters'].std()
            df_features['altitude_std'] = (df_features['altitude_mean_meters'] - altitude_mean) / altitude_std
            print(f"✅ Altitud estandarizada creada")
        
        # 2. Features de calidad sensorial combinadas
        sensory_cols = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
        existing_sensory = [col for col in sensory_cols if col in df_features.columns]
        
        if len(existing_sensory) >= 3:
            # Promedio de características sensoriales
            df_features['sensory_avg'] = df_features[existing_sensory].mean(axis=1)
            print(f"✅ Promedio sensorial creado")
            
            # Desviación estándar sensorial (consistencia)
            df_features['sensory_std'] = df_features[existing_sensory].std(axis=1)
            print(f"✅ Desviación estándar sensorial creada")
            
            # Mejor característica sensorial
            df_features['best_sensory'] = df_features[existing_sensory].max(axis=1)
            print(f"✅ Mejor característica sensorial creada")
        
        # 3. Features de defectos
        defect_cols = ['Category.One.Defects', 'Category.Two.Defects']
        existing_defects = [col for col in defect_cols if col in df_features.columns]
        
        if len(existing_defects) > 0:
            df_features['total_defects'] = df_features[existing_defects].sum(axis=1)
            print(f"✅ Total de defectos creado")
            
            # Indicador de café sin defectos
            df_features['no_defects'] = (df_features['total_defects'] == 0).astype(int)
            print(f"✅ Indicador sin defectos creado")
        
        # 4. Features de humedad
        if 'Moisture' in df_features.columns:
            # Categoría de humedad
            df_features['moisture_category'] = pd.cut(
                df_features['Moisture'],
                bins=[0, 0.10, 0.12, 0.15, float('inf')],
                labels=['Baja', 'Óptima', 'Aceptable', 'Alta']
            )
            print(f"✅ Categoría de humedad creada")
        
        # 5. Features de procesamiento
        if 'Processing.Method' in df_features.columns:
            # Simplificar métodos de procesamiento
            processing_mapping = {
                'Washed / Wet': 'Washed',
                'Natural / Dry': 'Natural',
                'Semi-washed / Semi-pulped': 'Honey',
                'Pulped Natural': 'Natural'
            }
            
            df_features['processing_simple'] = df_features['Processing.Method'].map(processing_mapping)
            df_features['processing_simple'] = df_features['processing_simple'].fillna(df_features['Processing.Method'])
            print(f"✅ Método de procesamiento simplificado")
        
        self.df = df_features
        return df_features
    
    def identify_feature_types(self):
        """
        Identifica y clasifica las features por tipo.
        
        Returns:
            tuple: (categorical_features, numerical_features)
        """
        if self.df is None:
            print("❌ No hay datos cargados")
            return None, None
        
        # Excluir variable objetivo
        df_features = self.df.drop(columns=['Total.Cup.Points'])
        
        # Identificar variables categóricas
        categorical_features = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Identificar variables numéricas
        numerical_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\n📋 CLASIFICACIÓN DE FEATURES")
        print(f"="*30)
        print(f"🏷️ Variables categóricas ({len(categorical_features)}):")
        for col in categorical_features:
            print(f"   - {col}")
        
        print(f"\n📊 Variables numéricas ({len(numerical_features)}):")
        for col in numerical_features:
            print(f"   - {col}")
        
        return categorical_features, numerical_features
    
    def create_preprocessor(self, categorical_features, numerical_features):
        """
        Crea el preprocesador para transformación de variables.
        
        Args:
            categorical_features (list): Lista de variables categóricas
            numerical_features (list): Lista de variables numéricas
            
        Returns:
            ColumnTransformer: Preprocesador configurado
        """
        print(f"\n🔧 CREANDO PREPROCESADOR")
        print(f"="*30)
        
        # Pipeline para variables numéricas (escalado estándar)
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Pipeline para variables categóricas (OneHot Encoding)
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combinar preprocesadores
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        self.preprocessor = preprocessor
        print(f"✅ Preprocesador creado")
        print(f"   - Variables numéricas: {len(numerical_features)}")
        print(f"   - Variables categóricas: {len(categorical_features)}")
        
        return preprocessor
    
    def prepare_data_for_training(self, test_size=0.2, random_state=42):
        """
        Prepara los datos finales para entrenamiento.
        
        Args:
            test_size (float): Proporción de datos para prueba
            random_state (int): Semilla aleatoria
            
        Returns:
            dict: Diccionario con datos preparados
        """
        if self.df is None:
            print("❌ No hay datos cargados")
            return None
        
        print(f"\n📊 PREPARANDO DATOS PARA ENTRENAMIENTO")
        print(f"="*45)
        
        # Separar features y target
        X = self.df.drop(columns=['Total.Cup.Points'])
        y = self.df['Total.Cup.Points']
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"📊 División de datos:")
        print(f"   - Entrenamiento: {X_train.shape} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   - Prueba: {X_test.shape} ({len(X_test)/len(X)*100:.1f}%)")
        
        # Identificar tipos de features
        categorical_features, numerical_features = self.identify_feature_types()
        
        # Crear preprocesador
        preprocessor = self.create_preprocessor(categorical_features, numerical_features)
        
        # Aplicar preprocesamiento
        print(f"\n🔄 APLICANDO PREPROCESAMIENTO")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Guardar nombres de features
        self.feature_names = preprocessor.get_feature_names_out()
        
        print(f"✅ Datos preprocesados:")
        print(f"   - Train: {X_train_processed.shape}")
        print(f"   - Test: {X_test_processed.shape}")
        print(f"   - Features totales: {len(self.feature_names)}")
        
        # Preparar diccionario de resultados
        data_dict = {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_original': X_train,
            'X_test_original': X_test,
            'feature_names': self.feature_names,
            'preprocessor': preprocessor,
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'data_shape': self.df.shape
        }
        
        return data_dict
    
    def save_processed_data(self, data_dict):
        """
        Guarda los datos procesados para el entrenamiento.
        
        Args:
            data_dict (dict): Diccionario con datos procesados
        """
        print(f"\n💾 GUARDANDO DATOS PROCESADOS")
        print(f"="*35)
        
        # 1. Guardar datos procesados como CSV (para facilidad de uso)
        processed_data_path = os.path.join(self.output_dir, 'processed_data.csv')
        
        # Combinar X_train_processed con y_train para guardar
        train_df = pd.DataFrame(data_dict['X_train'], columns=data_dict['feature_names'])
        train_df['Total.Cup.Points'] = data_dict['y_train'].values
        train_df['dataset'] = 'train'
        
        # Combinar X_test_processed con y_test
        test_df = pd.DataFrame(data_dict['X_test'], columns=data_dict['feature_names'])
        test_df['Total.Cup.Points'] = data_dict['y_test'].values
        test_df['dataset'] = 'test'
        
        # Unir y guardar
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        full_df.to_csv(processed_data_path, index=False)
        
        print(f"✅ Datos procesados guardados: {processed_data_path}")
        print(f"   - Total de registros: {len(full_df)}")
        print(f"   - Features: {len(data_dict['feature_names'])}")
        
        # 2. Guardar preprocesador
        import joblib
        preprocessor_path = os.path.join(self.output_dir, 'preprocessor.pkl')
        joblib.dump(data_dict['preprocessor'], preprocessor_path)
        print(f"✅ Preprocesador guardado: {preprocessor_path}")
        
        # 3. Guardar metadatos
        metadata = {
            'data_shape': data_dict['data_shape'],
            'feature_names': data_dict['feature_names'].tolist(),
            'categorical_features': data_dict['categorical_features'],
            'numerical_features': data_dict['numerical_features'],
            'train_size': len(data_dict['y_train']),
            'test_size': len(data_dict['y_test']),
            'target_stats': {
                'min': data_dict['y_train'].min(),
                'max': data_dict['y_train'].max(),
                'mean': data_dict['y_train'].mean(),
                'std': data_dict['y_train'].std()
            }
        }
        
        metadata_path = os.path.join(self.output_dir, 'metadata.pkl')
        joblib.dump(metadata, metadata_path)
        print(f"✅ Metadatos guardados: {metadata_path}")
        
        return processed_data_path
    
    def run_full_pipeline(self):
        """
        Ejecuta el pipeline completo de feature engineering.
        
        Returns:
            str: Ruta al archivo de datos procesados
        """
        print("🚀 INICIANDO PIPELINE COMPLETO DE FEATURE ENGINEERING")
        print("="*60)
        
        # 1. Cargar datos
        if not self.load_data():
            return None
        
        # 2. Analizar datos
        self.analyze_data()
        
        # 3. Limpiar datos
        self.clean_data()
        
        # 4. Crear features
        self.create_features()
        
        # 5. Preparar datos para entrenamiento
        data_dict = self.prepare_data_for_training()
        
        if data_dict is None:
            return None
        
        # 6. Guardar datos procesados
        processed_path = self.save_processed_data(data_dict)
        
        print(f"\n✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print(f"📁 Datos procesados guardados en: {processed_path}")
        
        return processed_path

if __name__ == "__main__":
    # Ejemplo de uso
    data_path = "../../data/processed/coffee_data_cleaned_final.csv"
    output_dir = "../../models/prediction"
    
    # Corregir rutas relativas
    import sys
    import os
    
    # Obtener el directorio actual del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    data_path = os.path.join(project_root, "data", "processed", "coffee_data_cleaned_final.csv")
    output_dir = os.path.join(project_root, "models", "prediction")
    
    print(f"📁 Ruta corregida de datos: {data_path}")
    print(f"📁 Ruta corregida de salida: {output_dir}")
    
    engineer = CoffeeFeatureEngineer(data_path, output_dir)
    processed_path = engineer.run_full_pipeline()
    
    if processed_path:
        print(f"\n🎉 Feature engineering completado!")
        print(f"📁 Usa el archivo {processed_path} para entrenar modelos")
