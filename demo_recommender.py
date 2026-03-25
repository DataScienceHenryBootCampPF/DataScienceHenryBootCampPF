import pandas as pd
import os

def ejecutar_demo():
    print("\n☕ BIENVENIDO AL BUSCADOR DE CAFÉS PREMIUM ☕")
    print("-" * 55) 
    
    try:
        csv_path = './models/recommender/coffee_flavor_segments.csv'
        if not os.path.exists(csv_path):
            print("❌ Error: No existe el archivo de segmentos. Ejecuta el Profiler primero.")
            return

        df = pd.read_csv(csv_path)
        perfiles = sorted(list(df['Perfil_Nombre'].unique()))
        
        # 1. Selección de Perfil
        print("Paso 1: Selecciona un perfil de sabor:")
        for i, p in enumerate(perfiles, 1):
            print(f"  {i}. {p}")
            
        sel_perfil = input("\nNúmero del perfil: ")

        if sel_perfil.isdigit() and 0 < int(sel_perfil) <= len(perfiles):
            perfil_buscado = perfiles[int(sel_perfil) - 1]
            
            # Filtramos países para ese perfil
            df_perfil = df[df['Perfil_Nombre'] == perfil_buscado]
            paises = sorted(df_perfil['Country.of.Origin'].unique())

            # 2. Selección de País
            print(f"\n🌍 Paso 2: Selecciona un país con perfil '{perfil_buscado}':")
            for i, pais in enumerate(paises, 1):
                print(f"  {i}. {pais}")
            print(f"  {len(paises) + 1}. VER TODOS LOS PAÍSES")
            
            sel_pais = input("\nNúmero del país: ")

            # 3. Lógica de Filtrado
            resultados = df_perfil.copy()
            if sel_pais.isdigit():
                idx_pais = int(sel_pais) - 1
                if 0 <= idx_pais < len(paises):
                    pais_elegido = paises[idx_pais]
                    resultados = resultados[resultados['Country.of.Origin'] == pais_elegido]
                    print(f"\n🔍 Buscando los mejores de {pais_elegido}...")
                elif idx_pais == len(paises):
                    print(f"\n🔍 Mostrando resultados globales para '{perfil_buscado}'...")
                else:
                    print("⚠️ Número no válido. Mostrando todos.")
            
            if resultados.empty:
                print("❌ No se encontraron resultados.")
            else:
                final_top = (resultados
                             .sort_values(by='Total.Cup.Points', ascending=False)
                             .drop_duplicates(subset=['Nombre_Comercial'])
                             [['Nombre_Comercial', 'Total.Cup.Points', 'Categoria_Calidad']]
                             .head(5))

                print(f"\n🏆 TOP 5 CAFÉS RECOMENDADOS:")
                display_df = final_top.reset_index(drop=True)
                display_df.index = display_df.index + 1
                display_df.columns = ['Marca', 'Puntaje', 'Categoría']
                
                print(display_df.to_string())
                print(f"💡 Perfil sensorial seleccionado: {perfil_buscado}")
        else:
            print("❌ Selección de perfil no válida.")

    except Exception as e:
        print(f"❌ Ocurrió un error: {e}")

if __name__ == "__main__":
    ejecutar_demo()