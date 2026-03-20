import pandas as pd
import numpy as np 

def recomendar_por_perfil(nombre_perfil, top_n=20):
    
    try:
        df = pd.read_csv('./models/recommender/coffee_flavor_segments.csv')
    except FileNotFoundError:
        return "Error: No se encuentra el archivo de segmentos. Ejecuta el pipeline primero."

    # 1. Filtrar por el nombre del perfil (Cluster)
    mask = df['Perfil_Nombre'].str.lower() == nombre_perfil.lower()
    candidatos = df[mask].copy()

    if candidatos.empty:
        opciones = df['Perfil_Nombre'].unique()
        return f"No se encontró el perfil. Intenta con: {list(opciones)}"

    # --- NUEVA LÓGICA DE CLASIFICACIÓN (Basada en tu análisis de Mediana) ---
    umbral_mediana = 82.5
    candidatos['Categoria_Calidad'] = np.where(
        candidatos['Total.Cup.Points'] >= umbral_mediana, 
        'Premium', 
        'Estándar'
    )

    # 2. Ordenar por puntaje de mayor a menor
    recomendaciones = candidatos.sort_values(by='Total.Cup.Points', ascending=False)

    # 3. Retornar las columnas incluyendo la nueva clasificación
    return recomendaciones[[
        'Nombre_Comercial', 
        'Total.Cup.Points', 
        'Categoria_Calidad', # <--- Agregada a la salida
        'Country.of.Origin'
    ]]