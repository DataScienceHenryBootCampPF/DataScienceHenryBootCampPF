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

    def _auto_label_clusters(self, df_arabica):
        """Asigna nombres basados en el ranking de atributos sensoriales."""
        resumen = df_arabica.groupby('Cluster')[self.features_sabor].mean()
        nombres = {}
        
        # 1. El de más acidez es 'Vibrante y Cítrico'
        cluster_vibrante = resumen['Acidity'].idxmax()
        nombres[cluster_vibrante] = "Vibrante y Cítrico"
        
        # 2. El de más cuerpo (de los restantes) es 'Cuerpo e Intensidad'
        restantes_body = resumen.drop(cluster_vibrante)
        cluster_cuerpo = restantes_body['Body'].idxmax()
        nombres[cluster_cuerpo] = "Cuerpo e Intensidad"
        
        # 3. El restante es 'Equilibrado y Dulce'
        ultimo_cluster = restantes_body.drop(cluster_cuerpo).index[0]
        nombres[ultimo_cluster] = "Equilibrado y Dulce"
        
        return nombres

    def run_pipeline(self, df):
        print("🚀 Iniciando Pipeline de Segmentación...")
        
        # 1. Limpieza
        df_clean = df.dropna(subset=self.features_sabor).copy()

        # 2. Separación
        df_arabica = df_clean[df_clean['Species'] == 'Arabica'].copy()
        df_robusta = df_clean[df_clean['Species'] == 'Robusta'].copy()

        # 3. Clustering Arabica
        X_scaled = self.scaler.fit_transform(df_arabica[self.features_sabor])
        df_arabica['Cluster'] = self.kmeans.fit_predict(X_scaled)

        # 4. Etiquetado
        perfiles_nombres = self._auto_label_clusters(df_arabica)
        df_arabica['Perfil_Nombre'] = df_arabica['Cluster'].map(perfiles_nombres)

        # 5. Robusta (Perfil Fijo)
        df_robusta['Cluster'] = 99 
        df_robusta['Perfil_Nombre'] = "Cuerpo Intenso (Robusta)"

        # 6. Consolidación y Categoría de Calidad (LA SOLUCIÓN AL ERROR)
        df_final = pd.concat([df_arabica, df_robusta]).reset_index(drop=True)
        
        umbral_mediana = 82.5
        df_final['Categoria_Calidad'] = np.where(
            df_final['Total.Cup.Points'] >= umbral_mediana, 
            'Premium', 
            'Estándar'
        )

        # 7. Guardar Artefactos
        self._save_artifacts(df_final)
        return df_final

    def _save_artifacts(self, df_final):
        os.makedirs(self.save_path, exist_ok=True)
        joblib.dump(self.kmeans, f"{self.save_path}kmeans_flavor.pkl")
        joblib.dump(self.scaler, f"{self.save_path}scaler_flavor.pkl")
        df_final.to_csv(f"{self.save_path}coffee_flavor_segments.csv", index=False)
        print(f"✅ CSV generado con Categoria_Calidad en: {self.save_path}")

if __name__ == "__main__":
    # Asegurate de que esta ruta sea la correcta en tu proyecto
    data_path = './data/processed/coffee_data_cleaned_final.csv'
    if os.path.exists(data_path):
        df_input = pd.read_csv(data_path)
        profiler = CoffeeFlavorProfiler()
        profiler.run_pipeline(df_input)
    else:
        print(f"❌ No se encontró el dataset en {data_path}")