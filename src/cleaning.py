import pandas as pd
import numpy as np

# --- FUNCIONES AUXILIARES (Deben definirse PRIMERO) ---

def _estandarizar_unidades(row):
    """Función auxiliar para convertir pies (ft) a metros (m)."""
    # Usamos .get() para evitar errores si la columna no existe en esa fila
    unidad = str(row.get('unit_of_measurement', '')).lower().strip()
    
    if 'ft' in unidad or 'feet' in unidad:
        cols = ['altitude_low_meters', 'altitude_high_meters', 'altitude_mean_meters']
        for col in cols:
            if col in row and pd.notnull(row[col]):
                # Modificamos el valor directamente en la fila
                row[col] = row[col] * 0.3048
    return row

def _categorizar_altitud_pro(row):
    """Asigna categorías de cultivo basadas en la especie y la altitud media."""
    metros = row.get('altitude_mean_meters')
    especie = row.get('Species')
    
    # Manejo de nulos o valores incoherentes
    if pd.isnull(metros) or metros <= 0:
        return 'No Especificado'
    
    # Lógica por especie (Crucial para tu PF)
    if especie == 'Arabica':
        if metros < 1000: return 'Baja (Arabica)'
        if 1000 <= metros < 1500: return 'Media (Arabica)'
        if 1500 <= metros < 1800: return 'Alta (Arabica)'
        return 'Premium (Arabica)'
    else: # Robusta
        if metros < 400: return 'Baja (Robusta)'
        if 400 <= metros < 700: return 'Media (Robusta)'
        return 'Alta (Robusta)'

# --- FUNCIÓN PRINCIPAL ---

def clean_coffee_pipeline(df):
    df_limpio = df.copy()

    # 1. Eliminación de columnas (incluyendo las de varianza cero)
    cols_to_drop = [
        'Lot.Number', 'Farm.Name', 'Mill', 'Producer', 'Company', 
        'ICO.Number', 'Owner.1', 'Quakers', 'Expiration', 
        'Certification.Body', 'Unnamed: 0', 'Harvest.Year', 
        'Certification.Address', 'Certification.Contact', 'Altitude',
        'Uniformity', 'Clean.Cup', 'Sweetness'
    ]
    df_limpio = df_limpio.drop(columns=cols_to_drop, errors='ignore')

    # 1.5 Filtrado de registros en cero (Puntajes "rotos")
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

    # 4. Imputación avanzada (Moda)
    cols_moda = ['Color', 'Variety', 'Processing.Method']
    for col in cols_moda:
        if col in df_limpio.columns:
            moda_por_grupo = df_limpio.groupby(['Country.of.Origin', 'Species'])[col].transform(
                lambda x: x.mode()[0] if not x.mode().empty else "Other"
            )
            df_limpio[col] = df_limpio[col].fillna(moda_por_grupo)

    # 5. Estandarización de unidades (Aplicamos la función auxiliar)
    df_limpio = df_limpio.apply(_estandarizar_unidades, axis=1)
    
    # Limpiamos columnas de apoyo de altitud
    df_aux_alt = ['altitude_low_meters', 'altitude_high_meters', 'unit_of_measurement']
    df_limpio = df_limpio.drop(columns=df_aux_alt, errors='ignore')

    # 6. Imputación de Altitud Media (Mediana)
    df_limpio['altitude_mean_meters'] = df_limpio['altitude_mean_meters'].fillna(
        df_limpio.groupby(['Country.of.Origin', 'Species'])['altitude_mean_meters'].transform('median')
    )
    df_limpio['altitude_mean_meters'] = df_limpio['altitude_mean_meters'].fillna(
        df_limpio.groupby('Species')['altitude_mean_meters'].transform('median')
    )

    # 7. Categorización (Aplicamos la función auxiliar)
    # Aquí es donde fallaba: nos aseguramos que axis=1 esté presente
    df_limpio['categoria_altitud'] = df_limpio.apply(_categorizar_altitud_pro, axis=1)

    return df_limpio