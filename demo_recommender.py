from src.recommender_engine import recomendar_por_perfil
import pandas as pd

def ejecutar_demo():
    print("☕ BIENVENIDO AL BUSCADOR DE CAFÉS PREMIUM ☕")
    print("-" * 45)
    
    # Mostrar opciones disponibles al usuario
    try:
        df = pd.read_csv('./models/recommender/coffee_flavor_segments.csv')
        perfiles = df['Perfil_Nombre'].unique()
        print("Perfiles disponibles para elegir:")
        for i, p in enumerate(perfiles, 1):
            print(f"{i}. {p}")
    except:
        print("⚠️ Advertencia: No se pudo cargar la lista de perfiles.")

    print("-" * 45)
    perfil_buscado = input("Escribe el nombre del perfil que te interesa: ")

    print(f"\n🔍 Buscando los mejores cafés '{perfil_buscado}'...")
    resultados = recomendar_por_perfil(perfil_buscado)

    if isinstance(resultados, str):
        print(resultados)
    else:
        print("\n✅ TOP 5 CAFÉS RECOMENDADOS POR PUNTAJE:")
        # Resetear el index solo para que se vea del 1 al 5 en la tabla
        print(resultados.reset_index(drop=True).rename(index=lambda x: x + 1))

if __name__ == "__main__":
    ejecutar_demo()