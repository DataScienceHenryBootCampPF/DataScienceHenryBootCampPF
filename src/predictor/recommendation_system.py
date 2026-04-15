"""
Sistema Híbrido de Recomendación de Café
========================================

Este módulo implementa un sistema de recomendación basado en similitud de coseno
utilizando 6 variables clave sensoriales.

Variables de Similitud:
- Aroma
- Flavor
- Aftertaste
- Acidity
- Body
- Balance

El sistema completa los valores no especificados por el usuario con el promedio
del dataset, calcula la similitud con todos los cafés disponibles, selecciona
los más similares y finalmente los ordena por puntuación total (calidad).

Author: Data Science Henry Bootcamp
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class HybridCoffeeRecommendationSystem:
    """
    Sistema híbrido de recomendación basado en similitud coseno.
    
    Características:
    - Utiliza 6 variables sensoriales clave.
    - Completa inputs faltantes con promedios del dataset.
    - Normaliza datos con un StandardScaler específico.
    - Ordena resultados finales por calidad (Total.Cup.Points).
    """
    
    SIMILARITY_FEATURES = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
    
    def __init__(self, data_path='data/processed/coffee_data_cleaned_final.csv', 
                 scaler_path='models/prediction/scaler_recommendation.pkl', save_scaler=True):
        """
        Inicializa el sistema de recomendación.
        
        Args:
            data_path (str): Ruta al archivo CSV de datos.
            scaler_path (str): Ruta donde guardar/cargar el StandardScaler.
            save_scaler (bool): Si es True, guarda el scaler si es nuevo.
        """
        self.data_path = data_path
        self.scaler_path = scaler_path
        self.save_scaler = save_scaler
        self.df = None
        self.scaler = None
        self.feature_means = {}
        self.df_normalized = None
        
        # Cargar datos e inicializar componentes
        self._load_data()
        self._calculate_feature_means()
        self._initialize_scaler()
        self._normalize_dataset()
        
        print(f"✨ Sistema de Recomendación Inicializado")
        print(f"   - Features: {self.SIMILARITY_FEATURES}")
        print(f"   - Total Cafés: {len(self.df)}")
    
    def _load_data(self):
        """Carga el dataset desde el archivo CSV."""
        try:
            if not os.path.exists(self.data_path):
                # Intentar ruta absoluta si relativa falla
                # Asumiendo estructura de proyecto estándar si se ejecuta desde raíz o src
                possible_paths = [
                    self.data_path,
                    os.path.join(os.getcwd(), self.data_path),
                    os.path.join(os.path.dirname(__file__), '../../..', self.data_path)
                ]
                
                found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        self.data_path = path
                        found = True
                        break
                
                if not found:
                    raise FileNotFoundError(f"No se encontró el archivo: {self.data_path}")

            self.df = pd.read_csv(self.data_path)
            
            # Verificar columnas necesarias
            required_cols = self.SIMILARITY_FEATURES + ['Total.Cup.Points', 'Species']
            missing = [col for col in required_cols if col not in self.df.columns]
            if missing:
                raise ValueError(f"Faltan columnas necesarias en el CSV: {missing}")
                
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            raise

    def _calculate_feature_means(self):
        """Calcula los promedios de las 6 variables de similitud."""
        means = self.df[self.SIMILARITY_FEATURES].mean()
        self.feature_means = means.to_dict()
        # print(f"📊 Promedios calculados: {self.feature_means}")

    def _initialize_scaler(self):
        """Crea o carga el StandardScaler específico."""
        try:
            # Verificar directorios
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                # print(f"✅ Scaler cargado de {self.scaler_path}")
            else:
                # Crear nuevo scaler
                self.scaler = StandardScaler()
                self.scaler.fit(self.df[self.SIMILARITY_FEATURES])
                
                if self.save_scaler:
                    joblib.dump(self.scaler, self.scaler_path)
                    print(f"💾 Nuevo scaler guardado en {self.scaler_path}")
                    
        except Exception as e:
            print(f"⚠️ Error con scaler (usando temporal): {e}")
            self.scaler = StandardScaler()
            self.scaler.fit(self.df[self.SIMILARITY_FEATURES])

    def _normalize_dataset(self):
        """Normaliza el dataset completo una sola vez para eficiencia."""
        self.matrix_normalized = self.scaler.transform(self.df[self.SIMILARITY_FEATURES])

    def _validar_entrada(self, kwargs):
        """
        Valida los inputs del usuario.
        Retorna (bool, str): (Es válido, Mensaje de error)
        """
        # 1. Verificar que al menos una feature de similitud esté presente
        present_features = [k for k in kwargs.keys() if k in self.SIMILARITY_FEATURES and kwargs[k] is not None]
        if not present_features:
            return False, "Debe especificar al menos una variable (Aroma, Flavor, etc.)"
        
        # 2. Validar rangos (soft validation, solo advertencia o error si es extremo)
        for feat in present_features:
            val = kwargs[feat]
            if not isinstance(val, (int, float)):
                 return False, f"El valor de {feat} debe ser numérico"
            if val < 0 or val > 10: # Rango extendido por si acaso, típicamente 5-10
                 return False, f"El valor de {feat} debe estar entre 0 y 10"
                 
        return True, ""

    def _completar_con_promedios(self, kwargs):
        """Construye el vector de entrada completando faltantes con promedios."""
        input_vector = []
        
        for feature in self.SIMILARITY_FEATURES:
            if feature in kwargs and kwargs[feature] is not None:
                input_vector.append(float(kwargs[feature]))
            else:
                input_vector.append(self.feature_means[feature])
                
        return np.array([input_vector])

    def recomendar(self, Flavor=None, Aftertaste=None, Aroma=None,
                   Acidity=None, Body=None, Balance=None, 
                   species=None, top_n=10):
        """
        Genera recomendaciones de café basadas en similitud.
        
        Args:
            Flavor, Aftertaste, Aroma, Acidity, Body, Balance (float): Valores 0-10.
            species (str): Filtro opcional ('Arabica', 'Robusta').
            top_n (int): Cantidad de recomendaciones.
            
        Returns:
            pd.DataFrame: Top N cafés recomendados.
        """
        inputs = {
            'Flavor': Flavor, 'Aftertaste': Aftertaste, 'Aroma': Aroma,
            'Acidity': Acidity, 'Body': Body, 'Balance': Balance
        }
        
        # 1. Validar
        es_valido, mensaje = self._validar_entrada(inputs)
        if not es_valido:
            print(f"❌ Error de validación: {mensaje}")
            return None
        
        # 2. Completar vector
        user_vector = self._completar_con_promedios(inputs)
        
        # 3. Normalizar vector usuario
        user_vector_scaled = self.scaler.transform(user_vector)
        
        # 4. Calcular similitud coseno
        # Calcula similitud entre el vector usuario y TODOS los cafés en la matriz normalizada
        similarities = cosine_similarity(user_vector_scaled, self.matrix_normalized)[0]
        
        # Añadir similitudes temporalmente al df (trabajar en copia para no afectar self.df)
        df_results = self.df.copy()
        df_results['similarity_score'] = similarities
        
        # 5. Aplicar filtros (si aplica)
        if species:
            df_results = df_results[df_results['Species'].str.lower() == species.lower()]
            if len(df_results) == 0:
                print(f"⚠️ No se encontraron cafés de la especie '{species}'")
                return pd.DataFrame()

        # 6. Obtener Top N por similitud
        # Ordenamos descendente por similitud y tomamos top_n
        top_similar = df_results.sort_values(by='similarity_score', ascending=False).head(top_n)
        
        # 7. Re-ordenar Top N por Total.Cup.Points (Calidad)
        final_recommendations = top_similar.sort_values(by='Total.Cup.Points', ascending=False)
        
        # Seleccionar columnas de interés para el retorno
        cols_to_return = [
            'Owner', 'Country.of.Origin', 'Region', 'Variety', 'Processing.Method', 'Species',
            'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 
            'Total.Cup.Points', 'similarity_score'
        ]
        
        # Asegurar que columnas existan (por si acaso Owner o Region son nulos/no existen)
        cols_exist = [c for c in cols_to_return if c in final_recommendations.columns]
        
        return final_recommendations[cols_exist]

if __name__ == "__main__":
    # Test rápido al ejecutar directamente
    print("🔬 Ejecutando test rápido del módulo...")
    try:
        rec_sys = HybridCoffeeRecommendationSystem()
        res = rec_sys.recomendar(Flavor=8.5, Body=8.0)
        print("\n🏆 Top 3 Recomendaciones (Flavor=8.5, Body=8.0):")
        print(res[['Country.of.Origin', 'Total.Cup.Points', 'similarity_score']].head(3))
    except Exception as e:
        print(f"❌ Error en test rápido: {e}")
