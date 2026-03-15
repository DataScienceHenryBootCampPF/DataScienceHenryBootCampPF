import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

class CoffeeDataPreprocessor:
    """
    Clase para preprocesar datos de café para entrenamiento de modelos.
    Maneja codificación de variables categóricas, escalado y preparación de features.
    """
    
    def __init__(self, target_column='Total.Cup.Points', test_size=0.2, random_state=42):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.column_transformer = None
        self.feature_columns = None
        self.categorical_features = None
        self.numerical_features = None
        
    def identify_variable_types(self, df):
        """
        Identifica automáticamente variables categóricas y numéricas.
        Excluye columnas que no deben usarse como features.
        """
        # Columnas a excluir del modelado
        exclude_columns = [
            self.target_column, 'Unnamed: 0', 'Lot.Number', 'Farm.Name', 
            'Mill', 'Producer', 'Company', 'ICO.Number', 'Owner.1', 
            'Quakers', 'Expiration', 'Certification.Body', 
            'Certification.Address', 'Certification.Contact', 'Altitude',
            'Grading.Date', 'In.Country.Partner', 'Bag.Weight'
        ]
        
        # Filtrar columnas disponibles
        available_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Identificar tipos de variables
        self.categorical_features = []
        self.numerical_features = []
        
        for col in available_columns:
            if df[col].dtype == 'object':
                self.categorical_features.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                self.numerical_features.append(col)
                
        self.feature_columns = self.categorical_features + self.numerical_features
        print(f"Features categóricas ({len(self.categorical_features)}): {self.categorical_features}")
        print(f"Features numéricas ({len(self.numerical_features)}): {self.numerical_features}")
        
        return self.categorical_features, self.numerical_features
    
    def create_preprocessor(self, df):
        """
        Crea el preprocesador para variables categóricas y numéricas.
        """
        # Para variables categóricas con baja cardinalidad usamos OneHot
        # Para variables con alta cardinalidad usamos LabelEncoder
        low_cardinality_cats = []
        high_cardinality_cats = []
        
        for cat in self.categorical_features:
            # Contar valores únicos directamente del DataFrame
            unique_values = len(df[cat].unique())
            if unique_values <= 10:  # Umbral para cardinalidad baja
                low_cardinality_cats.append(cat)
            else:
                high_cardinality_cats.append(cat)
        
        # Crear transformador de columnas
        transformers = []
        
        # Variables numéricas - escalado estándar
        if self.numerical_features:
            transformers.append(('num', StandardScaler(), self.numerical_features))
        
        # Variables categóricas de baja cardinalidad - OneHot
        if low_cardinality_cats:
            transformers.append(('cat_low', OneHotEncoder(drop='first', handle_unknown='ignore'), low_cardinality_cats))
        
        # Variables categóricas de alta cardinalidad - LabelEncoder (aplicado después)
        self.high_cardinality_cats = high_cardinality_cats
        
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Descarta columnas no especificadas
        )
        
        return self.column_transformer
    
    def fit_label_encoders(self, df):
        """
        Ajusta label encoders para variables categóricas de alta cardinalidad.
        """
        for cat in self.categorical_features:
            self.label_encoders[cat] = LabelEncoder()
            # Manejar valores nulos
            temp_series = df[cat].fillna('Unknown')
            
            # Ajustar encoder
            self.label_encoders[cat].fit(temp_series)
            
            # Asegurar que 'Unknown' esté en las clases
            if 'Unknown' not in self.label_encoders[cat].classes_:
                # Si no está, añadirlo manualmente
                classes = list(self.label_encoders[cat].classes_)
                classes.append('Unknown')
                self.label_encoders[cat].classes_ = np.array(classes)
    
    def transform_categorical_features(self, df):
        """
        Aplica transformación a variables categóricas de alta cardinalidad.
        """
        df_transformed = df.copy()
        
        for cat in self.high_cardinality_cats:
            if cat in df_transformed.columns:
                # Manejar valores nulos
                df_transformed[cat] = df_transformed[cat].fillna('Unknown')
                
                # Manejar valores no vistos durante el entrenamiento
                unique_values = set(df_transformed[cat].unique())
                known_values = set(self.label_encoders[cat].classes_)
                unknown_values = unique_values - known_values
                
                if unknown_values:
                    # Reemplazar valores desconocidos con 'Unknown'
                    df_transformed[cat] = df_transformed[cat].apply(
                        lambda x: x if x in known_values else 'Unknown'
                    )
                
                # Aplicar LabelEncoder
                df_transformed[cat] = self.label_encoders[cat].transform(df_transformed[cat])
        
        return df_transformed
    
    def prepare_data(self, df):
        """
        Prepara completamente los datos para entrenamiento.
        """
        print("🔄 Iniciando preprocesamiento de datos...")
        
        # Identificar tipos de variables
        self.identify_variable_types(df)
        
        # Ajustar label encoders
        self.fit_label_encoders(df)
        
        # Crear preprocesador
        self.create_preprocessor(df)
        
        # Separar features y target
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Transformar variables categóricas de alta cardinalidad
        X = self.transform_categorical_features(X)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=None
        )
        
        # Ajustar y transformar datos de entrenamiento
        X_train_processed = self.column_transformer.fit_transform(X_train)
        
        # Transformar datos de prueba
        X_test_processed = self.column_transformer.transform(X_test)
        
        # Obtener nombres de features después del procesamiento
        feature_names = self.get_feature_names()
        
        print(f"✅ Preprocesamiento completado:")
        print(f"   - Dataset de entrenamiento: {X_train_processed.shape}")
        print(f"   - Dataset de prueba: {X_test_processed.shape}")
        print(f"   - Features finales: {len(feature_names)}")
        
        return X_train_processed, X_test_processed, y_train, y_test, feature_names
    
    def get_feature_names(self):
        """
        Obtiene los nombres de las features después del procesamiento.
        """
        feature_names = []
        
        # Features numéricas (escaladas)
        if self.numerical_features:
            feature_names.extend(self.numerical_features)
        
        # Features categóricas de baja cardinalidad (OneHot)
        if hasattr(self.column_transformer, 'named_transformers_'):
            for name, transformer, columns in self.column_transformer.transformers_:
                if name == 'cat_low':
                    # Obtener nombres de las columnas OneHot
                    if hasattr(transformer, 'get_feature_names_out'):
                        cat_features = transformer.get_feature_names_out(columns)
                        feature_names.extend(cat_features)
        
        # Features categóricas de alta cardinalidad (LabelEncoded)
        feature_names.extend(self.high_cardinality_cats)
        
        return feature_names
    
    def get_species_balance_info(self, df):
        """
        Analiza el balance de especies para considerar en el entrenamiento.
        """
        if 'Species' in df.columns:
            species_counts = df['Species'].value_counts()
            total_samples = len(df)
            
            print("\n📊 Análisis de balance de especies:")
            for species, count in species_counts.items():
                percentage = (count / total_samples) * 100
                print(f"   - {species}: {count} muestras ({percentage:.1f}%)")
            
            # Determinar si hay desbalance significativo
            max_percentage = species_counts.max() / total_samples * 100
            if max_percentage > 80:
                print("   ⚠️  Se detectó desbalance significativo de clases")
                return True
            else:
                print("   ✅ Balance de clases aceptable")
                return False
        
        return False
    
    def create_regression_categories(self, y):
        """
        Crea categorías a partir del puntaje total para análisis estratificado.
        Útil para manejar el desbalance en regresión.
        """
        # Definir categorías de calidad
        bins = [0, 80, 85, 90, 100]
        labels = ['Baja', 'Media', 'Alta', 'Premium']
        
        y_categories = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
        
        print("\n📈 Distribución de categorías de calidad:")
        for label in labels:
            count = (y_categories == label).sum()
            percentage = (count / len(y)) * 100
            print(f"   - {label}: {count} muestras ({percentage:.1f}%)")
        
        return y_categories
