import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def configurar_rutas():
    if os.path.basename(os.getcwd()) == 'notebooks':
        return ".."
    else:
        return "."

def mostrar_opciones_lista(lista):
    """Muestra una lista numerada."""
    for i, opcion in enumerate(lista):
        print(f"{i}: {opcion}")

def ejecutar_demo():
    prefix = configurar_rutas()
    path_data = os.path.join(prefix, "data", "processed", "coffee_data_for_training.csv")
    path_metrics = os.path.join(prefix, "metrics", "model_comparison_ranking.csv")
    path_models = os.path.join(prefix, "models", "prediction")

    if not os.path.exists(path_data) or not os.path.exists(path_metrics):
        print(f"❌ ERROR: Archivos no encontrados.")
        return

    try:
        df_train = pd.read_csv(path_data)
        df_results = pd.read_csv(path_metrics)
        
        best_model_name = df_results.iloc[0]['Model']
        mae_modelo = df_results.iloc[0]['MAE'] 
        best_model = joblib.load(os.path.join(path_models, f"coffee_model_{best_model_name}.pkl"))

        print("\n" + "="*60)
        print(f"☕ SIMULADOR DE CALIDAD - COFFEE PREDICTOR PRO")
        print(f"🤖 Motor: {best_model_name} | Error Medio: ±{mae_modelo:.2f}")
        print("="*60)

        inputs_usuario = {}

        paises = sorted(df_train['Country.of.Origin'].unique())
        print("\n--- Seleccione País de Origen ---")
        mostrar_opciones_lista(paises)
        pais_sel = paises[int(input("Número de País: "))]
        inputs_usuario['Country.of.Origin'] = [pais_sel]

        regiones = sorted(df_train[df_train['Country.of.Origin'] == pais_sel]['Region'].unique())
        print(f"\n--- Seleccione Región de {pais_sel} ---")
        mostrar_opciones_lista(regiones)
        region_sel = regiones[int(input("Número de Región: "))]
        inputs_usuario['Region'] = [region_sel]

        for col, nombre in {'Variety': 'Variedad', 'Processing.Method': 'Proceso', 'Color': 'Color'}.items():
            opciones = sorted(df_train[col].dropna().unique())
            print(f"\n--- Seleccione {nombre} ---")
            mostrar_opciones_lista(opciones)
            inputs_usuario[col] = [opciones[int(input(f"Número de {nombre}: "))]]

        df_geo = df_train[(df_train['Country.of.Origin'] == pais_sel) & (df_train['Region'] == region_sel)]
        alt_min, alt_max = df_geo['altitude_mean_meters'].min(), df_geo['altitude_mean_meters'].max()

        print(f"\n--- Altitud de la Finca ---")
        print(f"💡 En {region_sel}, el rango histórico es {alt_min:.0f}m - {alt_max:.0f}m")
        altitud = float(input(f"Ingrese altitud (msnm): "))

        data_final = pd.DataFrame({
            **inputs_usuario,
            'categoria_altitud': ['Alta' if altitud > 1200 else 'Media' if altitud > 800 else 'Baja'],
            'Moisture': [0.11],
            'Category.One.Defects': [0],
            'Category.Two.Defects': [0],
            'altitude_mean_meters': [altitud]
        })

        pred = best_model.predict(data_final)[0]
        
        umbral_mediana = 82.5
        limite_inf = pred - mae_modelo
        limite_sup = pred + mae_modelo

        print("\n" + "*"*40)
        print(f"📊 RESULTADO DE LA PREDICCIÓN")
        print(f"Puntaje Estimado: {pred:.2f}")
        print(f"Rango de Confianza (95%): {limite_inf:.2f} a {limite_sup:.2f}")
        print("-" * 40)
        
        if pred >= umbral_mediana:
            print(f"🏆 CATEGORÍA: PREMIUM")
            print(f"   (Puntaje por encima de la mediana del mercado)")
        else:
            print(f"📦 CATEGORÍA: ESTÁNDAR")
            print(f"   (Puntaje dentro del rango base de comercialización)")
        
        print("*"*40 + "\n")

    except Exception as e:
        print(f"❌ Error en la simulación: {e}")

if __name__ == "__main__":
    ejecutar_demo()