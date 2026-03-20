import pandas as pd

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

    # 2. Ordenar por puntaje de mayor a menor
    recomendaciones = candidatos.sort_values(by='Total.Cup.Points', ascending=False)

    # 3. Retornar las columnas (sin el .head(5) para que la demo pueda filtrar marcas repetidas)
    return recomendaciones[['Nombre_Comercial', 'Total.Cup.Points', 'Country.of.Origin']]