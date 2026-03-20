from src.recommender.recommender_engine import recomendar_por_perfil
import pandas as pd

def ejecutar_demo():
    print("\n☕ BIENVENIDO AL BUSCADOR DE CAFÉS PREMIUM ☕")
    print("-" * 55) # Un poco más ancho para la nueva columna
    
    try:
        df = pd.read_csv('./models/recommender/coffee_flavor_segments.csv')
        perfiles = list(df['Perfil_Nombre'].unique())
        
        print("Perfiles disponibles para elegir:")
        for i, p in enumerate(perfiles, 1):
            print(f"{i}. {p}")
            
        print("-" * 55)
        seleccion = input("Selecciona el NÚMERO del perfil que te interesa: ")

        if seleccion.isdigit():
            indice = int(seleccion) - 1
            
            if 0 <= indice < len(perfiles):
                perfil_buscado = perfiles[indice]
                print(f"\n🔍 Buscando los mejores cafés con perfil '{perfil_buscado}'...")
                
                resultados = recomendar_por_perfil(perfil_buscado)

                if isinstance(resultados, str):
                    print(resultados)
                else:
                    # AJUSTE AQUÍ: Incluimos 'Categoria_Calidad' en el filtrado
                    final_top = (resultados
                                 .drop_duplicates(subset=['Nombre_Comercial'])
                                 [['Nombre_Comercial', 'Total.Cup.Points', 'Categoria_Calidad']] # <--- Agregada
                                 .head(5))

                    print(f"\n TOP 5 MARCAS ÚNICAS EN '{perfil_buscado}':")
                    
                    display_df = final_top.reset_index(drop=True)
                    display_df.index = display_df.index + 1
                    
                    # AJUSTE AQUÍ: Renombramos las 3 columnas
                    display_df.columns = ['Marca', 'Puntaje', 'Categoría']
                    
                    print(display_df)
                    print("\n💡 Nota: La categoría 'Premium' se asigna a puntajes >= 82.5 (Mediana).")
            else:
                print(f"Error: El número {seleccion} no está en la lista.")
        else:
            print("Error: Por favor, ingresa solo el número (ej: 1).")

    except FileNotFoundError:
        print("Error: No se encontró el archivo de las recomendaciones.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    ejecutar_demo()