import pandas as pd
import numpy as np
import random
import os

# --- FUNCIONES AUXILIARES (Lógica interna) ---

def _generar_marca_comercial(row):
    """Crea la marca: Palabra fija por dueño + 3 iniciales del Dueño."""
    estilos = ['Mill', 'Private Reserve', 'Select Harvest', 'Heritage', 'Origins']
    owner = str(row.get('Owner', 'Unknown')).strip()
    iniciales = owner[:3].upper()
    random.seed(owner)
    txt = random.choice(estilos)
    return f"{txt} {iniciales}"

def _estandarizar_unidades(row):
    """Convierte pies (ft) a metros (m)."""
    unidad = str(row.get('unit_of_measurement', '')).lower().strip()
    if 'ft' in unidad or 'feet' in unidad:
        cols = ['altitude_low_meters', 'altitude_high_meters', 'altitude_mean_meters']
        for col in cols:
            if col in row and pd.notnull(row[col]):
                row[col] = row[col] * 0.3048
    return row

def _categorizar_altitud_pro(row):
    """Categoriza la altitud según la especie."""
    metros = row.get('altitude_mean_meters')
    especie = row.get('Species')
    if pd.isnull(metros) or metros <= 0:
        return 'No Especificado'
    
    if especie == 'Arabica':
        if metros < 1000: return 'Baja (Arabica)'
        if 1000 <= metros < 1500: return 'Media (Arabica)'
        if 1500 <= metros < 1800: return 'Alta (Arabica)'
        return 'Premium (Arabica)'
    else:
        if metros < 400: return 'Baja (Robusta)'
        if 400 <= metros < 700: return 'Media (Robusta)'
        return 'Alta (Robusta)'

# --- FUNCIÓN PRINCIPAL DEL PIPELINE ---

def clean_coffee_pipeline(df):
    """Limpia el DataFrame, realiza imputaciones y genera branding."""
    df_limpio = df.copy()

    # 1. Eliminación de columnas irrelevantes
    cols_to_drop = [
        'Lot.Number', 'Farm.Name', 'Mill', 'Producer', 'Company', 
        'ICO.Number', 'Owner.1', 'Quakers', 'Expiration', 
        'Certification.Body', 'Unnamed: 0', 'Harvest.Year', 
        'Certification.Address', 'Certification.Contact', 'Altitude',
        'Uniformity', 'Clean.Cup', 'Sweetness'
    ]
    df_limpio = df_limpio.drop(columns=cols_to_drop, errors='ignore')

    # 1.5 Filtrado de puntajes en cero (Limpieza de errores de carga)
    if 'Total.Cup.Points' in df_limpio.columns:
        df_limpio = df_limpio[df_limpio['Total.Cup.Points'] > 0]

    # 2. Imputaciones básicas
    df_limpio['Owner'] = df_limpio['Owner'].fillna('Desconocido')
    if 'Country.of.Origin' in df_limpio.columns:
        df_limpio.loc[df_limpio['Country.of.Origin'].isnull(), 'Country.of.Origin'] = 'United States'
    df_limpio['Region'] = df_limpio['Region'].fillna("Other")

    # 3. Normalización de Color y Procesamiento
    mapa_colores = {'Bluish-Green': 'Blue-Green', 'Blue-green': 'Blue-Green', 'Greenish': 'Green-ish', 'None': np.nan}
    df_limpio['Color'] = df_limpio['Color'].replace(mapa_colores)
    if 'Processing.Method' in df_limpio.columns:
        df_limpio['Processing.Method'] = df_limpio['Processing.Method'].str.strip().str.title()

    # 4. Imputación por Moda (Contextual por País y Especie)
    cols_moda = ['Color', 'Variety', 'Processing.Method']
    for col in cols_moda:
        if col in df_limpio.columns:
            moda_por_grupo = df_limpio.groupby(['Country.of.Origin', 'Species'])[col].transform(
                lambda x: x.mode()[0] if not x.mode().empty else "Other"
            )
            df_limpio[col] = df_limpio[col].fillna(moda_por_grupo)

    # 5. Estandarización de unidades (Pies a Metros)
    df_limpio = df_limpio.apply(_estandarizar_unidades, axis=1)
    
    # --- 5.1 LIMPIEZA DE ALTITUDES IMPOSIBLES (OUTLIERS) ---
    # Filtramos valores > 4000m pero MANTENEMOS los nulos para imputarlos luego
    if 'altitude_mean_meters' in df_limpio.columns:
        antes = len(df_limpio)
        # La condición dice: "Quedate con los que son <= 4000 O los que son nulos"
        df_limpio = df_limpio[(df_limpio['altitude_mean_meters'] <= 4000) | (df_limpio['altitude_mean_meters'].isna())]
        eliminados = antes - len(df_limpio)
        if eliminados > 0:
            print(f"🧹 Se eliminaron {eliminados} registros con altitudes espaciales (> 4000m).")

    df_aux_alt = ['altitude_low_meters', 'altitude_high_meters', 'unit_of_measurement']
    df_limpio = df_limpio.drop(columns=df_aux_alt, errors='ignore')

    # 6. IMPUTACIÓN DE ALTITUD (RESCATE DE REGISTROS NULOS)
    # Llenamos los 230 registros nulos usando la mediana por origen y especie
    df_limpio['altitude_mean_meters'] = df_limpio['altitude_mean_meters'].fillna(
        df_limpio.groupby(['Country.of.Origin', 'Species'])['altitude_mean_meters'].transform('median')
    )
    # Backup: si el grupo país/especie no tiene datos, usamos la especie general
    df_limpio['altitude_mean_meters'] = df_limpio['altitude_mean_meters'].fillna(
        df_limpio.groupby('Species')['altitude_mean_meters'].transform('median')
    )

    # 7. Categorización de Altitud (Lógica por Especie)
    df_limpio['categoria_altitud'] = df_limpio.apply(_categorizar_altitud_pro, axis=1)

    # 8. Generación de Marca Comercial (Branding)
    df_limpio['Nombre_Comercial'] = df_limpio.apply(_generar_marca_comercial, axis=1)

    return df_limpio


if __name__ == "__main__":
    ruta_raw = "./data/raw/merged_data_cleaned.csv"
    ruta_processed = "./data/processed/coffee_data_cleaned_final.csv"

    if os.path.exists(ruta_raw):
        print(f"⏳ Leyendo datos desde {ruta_raw}...")
        df_merged = pd.read_csv(ruta_raw)
        
        print("🛠️ Ejecutando pipeline de limpieza...")
        df_final = clean_coffee_pipeline(df_merged)
        
        # Guardar resultado
        os.makedirs(os.path.dirname(ruta_processed), exist_ok=True)
        df_final.to_csv(ruta_processed, index=False)
        
        print(f"✅ ¡Proceso terminado! Registros finales: {len(df_final)}")
        print(f"📁 Archivo guardado en: {ruta_processed}")
    else:
        print(f"❌ Error: No se encontró el archivo en {ruta_raw}")