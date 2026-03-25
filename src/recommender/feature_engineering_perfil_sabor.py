import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

class CoffeeFlavorProfiler:
    def __init__(self, n_clusters_arabica=3, save_path='./models/recommender/'):
        self.n_clusters = n_clusters_arabica
        self.save_path = save_path
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters_arabica, random_state=42, n_init=10)
        self.features_sabor = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
        self.perfiles_nombres = {}

    def _auto_label_clusters(self, df_arabica):
        """Asigna nombres únicos basados en el ranking de atributos para evitar duplicados."""
        resumen = df_arabica.groupby('Cluster')[self.features_sabor].mean()
        nombres = {}
        
        # 1. El que tenga el promedio más alto de Acidez se lleva 'Vibrante y Cítrico'
        cluster_vibrante = resumen['Acidity'].idxmax()
        nombres[cluster_vibrante] = "Vibrante y Cítrico"
        
        # 2. De los clusters que quedan, el que tenga más Body es 'Cuerpo e Intensidad'
        restantes_body = resumen.drop(cluster_vibrante)
        cluster_cuerpo = restantes_body['Body'].idxmax()
        nombres[cluster_cuerpo] = "Cuerpo e Intensidad"
        
        # 3. El último cluster restante se etiqueta como 'Equilibrado y Dulce'
        ultimo_cluster = restantes_body.drop(cluster_cuerpo).index[0]
        nombres[ultimo_cluster] = "Equilibrado y Dulce"
                
        return nombres

    def run_pipeline(self, df):

        # 1. Limpieza de nulos
        df_clean = df.dropna(subset=self.features_sabor).copy()

        # 2. Separación por especies
        df_arabica = df_clean[df_clean['Species'] == 'Arabica'].copy()
        df_robusta = df_clean[df_clean['Species'] == 'Robusta'].copy()

        # 3. Entrenamiento Arabica
        print(f"🧠 Entrenando {self.n_clusters} clusters para Arabica...")
        X_scaled = self.scaler.fit_transform(df_arabica[self.features_sabor])
        df_arabica['Cluster'] = self.kmeans.fit_predict(X_scaled)

        # 4. Etiquetado dinámico (Aquí aplicamos la nueva lógica)
        self.perfiles_nombres = self._auto_label_clusters(df_arabica)
        df_arabica['Perfil_Nombre'] = df_arabica['Cluster'].map(self.perfiles_nombres)

        # 5. Robusta como Perfil Fijo
        df_robusta['Cluster'] = 99 
        df_robusta['Perfil_Nombre'] = "Cuerpo Intenso (Robusta)"

        # 6. Consolidación
        df_final = pd.concat([df_arabica, df_robusta]).reset_index(drop=True)

        # 7. Guardar Artefactos
        self._save_artifacts(df_final)
        return df_final

    def _save_artifacts(self, df_final):
        os.makedirs(self.save_path, exist_ok=True)
        joblib.dump(self.kmeans, f"{self.save_path}kmeans_flavor.pkl")
        joblib.dump(self.scaler, f"{self.save_path}scaler_flavor.pkl")
        df_final.to_csv(f"{self.save_path}coffee_flavor_segments.csv", index=False)
        print(f"✅ Archivos guardados en: {self.save_path}")

if __name__ == "__main__":
    data_path = './data/processed/coffee_data_cleaned_final.csv'
    if os.path.exists(data_path):
        df_input = pd.read_csv(data_path)
        profiler = CoffeeFlavorProfiler()
        df_resultados = profiler.run_pipeline(df_input)
        
        print("\n Resumen de Segmentación Final (Sprint 1):")
        print("-" * 40)
        print(df_resultados['Perfil_Nombre'].value_counts())
    else:
        print(f"No se encontró el dataset en {data_path}")